pub use config::CONFIG;

use bincode::{Decode, Encode};
use fjall::{KvSeparationOptions, PartitionCreateOptions};
use serde::{Deserialize, Serialize};

mod config;

pub fn kv_sep_partition_option() -> PartitionCreateOptions {
    PartitionCreateOptions::default()
        .max_memtable_size(128_000_000)
        .with_kv_separation(
            KvSeparationOptions::default()
                .separation_threshold(750)
                .file_target_size(256_000_000),
        )
}

#[derive(Debug, Encode, Decode, Serialize, Deserialize)]
pub struct Case {
    #[serde(rename = "原始链接")]
    pub doc_id: String,
    #[serde(rename = "案号")]
    pub case_id: String,
    #[serde(rename = "案件名称")]
    pub case_name: String,
    #[serde(rename = "法院")]
    pub court: String,
    // #[serde(rename = "所属地区")]
    // pub region: String,
    #[serde(rename = "案件类型")]
    pub case_type: String,
    // #[serde(rename = "案件类型编码")]
    // case_type_code: String,
    // #[serde(rename = "来源")]
    // source: String,
    #[serde(rename = "审理程序")]
    pub procedure: String,
    #[serde(rename = "裁判日期")]
    pub judgment_date: String,
    #[serde(rename = "公开日期")]
    pub public_date: String,
    #[serde(rename = "当事人")]
    pub parties: String,
    #[serde(rename = "案由")]
    pub cause: String,
    #[serde(rename = "法律依据")]
    pub legal_basis: String,
    #[serde(rename = "全文")]
    pub full_text: String,
}
