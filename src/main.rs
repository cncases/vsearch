use bincode::config::standard;
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use fjall::{Config, PartitionCreateOptions};
use qdrant_client::{
    Qdrant,
    qdrant::{PointStruct, UpsertPointsBuilder},
};
use scraper::Html;
use serde_json::Map;
use tracing::info;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
use vsearch::{CONFIG, Case, kv_sep_partition_option};

// feature cuda
#[cfg(feature = "cuda")]
use ort::ep::{self, ArenaExtendStrategy};

#[tokio::main]
async fn main() {
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::new("info,ort=warn"))
        .with(tracing_subscriber::fmt::layer())
        .init();
    let client = Qdrant::from_url(&*CONFIG.qdrant_rpc).build().unwrap();

    let batch_size = CONFIG.batch_size.unwrap_or(64);
    info!("batch size: {}", batch_size);

    #[cfg(feature = "cuda")]
    let eps = {
        let cuda_ep = ep::CUDA::default()
            .with_tf32(true)
            .with_memory_limit(
                8 * 1024 * 1024 * 1024, // 8 GB
            )
            .with_arena_extend_strategy(ArenaExtendStrategy::SameAsRequested)
            .build();
        vec![cuda_ep]
    };
    #[cfg(not(feature = "cuda"))]
    let eps = vec![];

    let mut model = TextEmbedding::try_new(
        InitOptions::new(EmbeddingModel::BGESmallZHV15)
            .with_show_download_progress(true)
            .with_execution_providers(eps),
    )
    .unwrap();

    let keyspace = Config::new(CONFIG.db.as_str()).open().unwrap();
    let db = keyspace
        .open_partition("cases", kv_sep_partition_option())
        .unwrap();
    let progress_ks = keyspace
        .open_partition("progress", PartitionCreateOptions::default())
        .unwrap();

    let mut case_count = 0;
    let mut batch = 0;
    let mut lengths = Vec::with_capacity(batch_size);
    let mut documents = Vec::with_capacity(batch_size);
    let mut ids = Vec::with_capacity(batch_size);
    let now = std::time::Instant::now();

    let progress = if let Some(b) = progress_ks.get("progress").unwrap() {
        u32::from_be_bytes(b[..].try_into().unwrap())
    } else {
        0
    };

    for i in db.iter() {
        let (k, v) = i.unwrap();
        let id = u32::from_be_bytes(k[..].try_into().unwrap());
        if id % 10000 == 0 {
            info!("case count: {}, id: {}", case_count, id);
        }
        if id <= progress {
            continue;
        }

        let (case, _): (Case, _) = bincode::decode_from_slice(&v, standard()).unwrap();
        if case.case_type != "刑事案件" {
            continue;
        }
        if case.full_text.is_empty() {
            continue;
        }
        case_count += 1;

        let full_text = remove_html_tags(&case.full_text);
        let chunks = chunk_chinese_text_backward(&full_text, 512, 512);

        if chunks.len() == 1 {
            documents.extend(chunks);
            ids.push(id as u64);
        } else {
            info!("case {id} chunk len:{}", chunks.len());
            let mut tmp = Vec::with_capacity(chunks.len());
            for chunk in chunks {
                lengths.push(chunk.len()); // 字符长度
                tmp.push(chunk);
            }
            let embeddings = model.embed(&documents, None).unwrap();
            let mut avg_embedding = length_weighted_mean(&embeddings, &lengths);
            l2_normalize(&mut avg_embedding);
            upload_embeddings(vec![avg_embedding], &vec![id as u64], &client).await;
        }

        if documents.len() >= batch_size {
            let embeddings = model.embed(&documents, Some(batch_size)).unwrap();
            upload_embeddings(embeddings, &ids, &client).await;

            batch += 1;
            documents.clear();
            ids.clear();
            progress_ks.insert("progress", id.to_be_bytes()).unwrap();
            info!(
                "batch={batch}, case: {case_count}, time: {}, id: {id}",
                now.elapsed().as_secs()
            );
        }
    }

    if !documents.is_empty() {
        let embeddings = model.embed(&documents, Some(batch_size)).unwrap();
        upload_embeddings(embeddings, &ids, &client).await;
        batch += 1;
        documents.clear();
        ids.clear();
        info!(
            "batch={batch}, case: {case_count}, time: {}",
            now.elapsed().as_secs()
        );
    }

    info!(
        "all done: case: {case_count}, time: {}",
        now.elapsed().as_secs()
    );
}

fn remove_html_tags(html: &str) -> String {
    let document = Html::parse_document(html);
    document
        .root_element()
        .text()
        .collect::<Vec<_>>()
        .join("\n")
}

/// 粗略但工程上可靠：
/// - 汉字 / 标点：1 token
/// - ASCII：3 字符 ≈ 1 token
fn estimate_tokens(text: &str) -> usize {
    let mut tokens = 0;
    let mut ascii_run = 0;

    for ch in text.chars() {
        if ch.is_ascii() {
            ascii_run += 1;
            if ascii_run == 3 {
                tokens += 1;
                ascii_run = 0;
            }
        } else {
            if ascii_run > 0 {
                tokens += 1;
                ascii_run = 0;
            }
            tokens += 1;
        }
    }

    if ascii_run > 0 {
        tokens += 1;
    }

    tokens
}

fn split_by_punctuation<'a>(text: &'a str, seps: &[char]) -> Vec<&'a str> {
    text.split_inclusive(seps).collect()
}

pub fn chunk_chinese_text_backward(
    text: &str,
    target_tokens: usize,
    max_tokens: usize,
) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut buffer: Vec<&str> = Vec::new();
    let mut buffer_tokens = 0;

    // 第一层：段落
    for para in text.split("\n") {
        let para = para.trim();
        if para.is_empty() {
            continue;
        }

        let units = split_by_punctuation(para, &['。', '！', '？', '；', '\n']);
        for unit in units {
            let unit_tokens = estimate_tokens(&unit);

            // 直接塞
            buffer.push(unit);
            buffer_tokens += unit_tokens;

            // 没超 max，继续
            if buffer_tokens <= max_tokens {
                continue;
            }

            // 超了：开始倒逼
            // 从最后一个 unit 往回退
            let mut rollback = Vec::new();

            while buffer_tokens > target_tokens && buffer.len() > 1 {
                if let Some(last) = buffer.pop() {
                    let t = estimate_tokens(&last);
                    buffer_tokens -= t;
                    rollback.push(last);
                }
            }

            // flush 当前 buffer
            if !buffer.is_empty() {
                chunks.push(buffer.concat());
            }

            // 重建 buffer：把 rollback 放回
            buffer.clear();
            buffer_tokens = 0;

            for u in rollback.into_iter().rev() {
                let t = estimate_tokens(&u);
                buffer.push(u);
                buffer_tokens += t;
            }
        }
    }

    if !buffer.is_empty() {
        chunks.push(buffer.concat());
    }

    chunks
}

fn l2_normalize(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

async fn upload_embeddings(embeddings: Vec<Vec<f32>>, ids: &Vec<u64>, client: &Qdrant) {
    let mut points = Vec::with_capacity(embeddings.len());
    for (i, embedding) in embeddings.into_iter().enumerate() {
        let id = ids[i];
        let object = Map::new();
        let point = PointStruct::new(id, embedding, object);
        points.push(point);
    }
    client
        .upsert_points(UpsertPointsBuilder::new(&*CONFIG.collection_name, points))
        .await
        .unwrap();
}

fn length_weighted_mean(embeddings: &[Vec<f32>], lengths: &[usize]) -> Vec<f32> {
    let dim = embeddings[0].len();
    let mut result = vec![0.0f32; dim];
    let mut weight_sum = 0.0f32;

    for (emb, &len) in embeddings.iter().zip(lengths.iter()) {
        let w = len as f32;
        for i in 0..dim {
            result[i] += emb[i] * w;
        }
        weight_sum += w;
    }

    if weight_sum > 0.0 {
        for v in &mut result {
            *v /= weight_sum;
        }
    }

    result
}
