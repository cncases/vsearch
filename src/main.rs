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
    let mut documents = Vec::with_capacity(batch_size);
    let mut ids = Vec::with_capacity(batch_size);
    let now = std::time::Instant::now();

    let progress = if let Some(p) = CONFIG.progress {
        p
    } else {
        if let Some(b) = progress_ks.get("progress").unwrap() {
            u32::from_be_bytes(b[..].try_into().unwrap())
        } else {
            0
        }
    };

    info!("Progress: {progress}");

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
        documents.push(full_text);
        ids.push(id as u64);

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
