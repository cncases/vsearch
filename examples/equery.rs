use std::time::Duration;

use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use qdrant_client::{
    Qdrant,
    qdrant::{SearchParamsBuilder, SearchPointsBuilder, point_id::PointIdOptions},
};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
use vsearch::CONFIG;

#[tokio::main]
async fn main() {
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::new("info"))
        .with(tracing_subscriber::fmt::layer())
        .init();
    let client = Qdrant::from_url(&*CONFIG.qdrant_rpc).build().unwrap();
    let mut model = TextEmbedding::try_new(
        InitOptions::new(EmbeddingModel::BGESmallZHV15).with_show_download_progress(true),
    )
    .unwrap();

    let query_str = vec!["北京动物保护"];
    let query = model.embed(query_str, None).unwrap();

    let query_vec = query[0].clone();
    println!("{:?}", query_vec);

    let search_result = client
        .search_points(
            SearchPointsBuilder::new(&*CONFIG.collection_name, query_vec, 30)
                .with_payload(false)
                .params(SearchParamsBuilder::default().exact(true)),
        )
        .await
        .unwrap();

    for point in &search_result.result {
        let id = point
            .id
            .as_ref()
            .unwrap()
            .point_id_options
            .as_ref()
            .unwrap();
        match id {
            PointIdOptions::Num(id_num) => {
                let (case_id, chunk_id) = split_id(*id_num);
                println!(
                    "Point ID: {}, Case ID: {}, Chunk ID: {}",
                    id_num, case_id, chunk_id
                );
            }
            PointIdOptions::Uuid(uuid) => {
                println!("Point UUID: {}", uuid);
            }
        }
    }
}

fn split_id(id: u64) -> (u32, u32) {
    let case_id = (id >> 32) as u32;
    let chunk_id = (id & 0xFFFF_FFFF) as u32;
    (case_id, chunk_id)
}
