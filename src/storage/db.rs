use crate::storage::schema::{
    InsertCostOptimizationsTable, InsertLogsTable, InsertQualityMetricsTable,
    SelectCostOptimizationsTable, SelectLogsTable, SentinelSchema,
};
use chrono::{DateTime, Utc};
use drizzle::core::OrderBy;
use drizzle::core::expr::eq;
use drizzle::sqlite::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use uuid::Uuid;

pub type DrizzleDb = drizzle::sqlite::rusqlite::Drizzle<SentinelSchema>;
pub type SharedDrizzleDb = Arc<Mutex<DrizzleDb>>;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RequestLog {
    pub id: Uuid,
    pub timestamp: DateTime<Utc>,
    pub provider: String,
    pub model: String,
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub latency_ms: u32,
    pub cost_usd: f64,
    pub cost_saved: f64,
    pub quality_score: f64,
    pub cache_hit: bool,
    pub pii_redacted: bool,
    pub status: String,
    pub error_message: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct CostOptimization {
    pub id: Uuid,
    pub timestamp: DateTime<Utc>,
    pub endpoint_pattern: String,
    pub original_model: String,
    pub suggested_model: String,
    pub potential_savings_percent: f64,
    pub confidence_score: f64,
    pub reason: String,
    pub status: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct QualityMetric {
    pub id: Uuid,
    pub timestamp: DateTime<Utc>,
    pub model: String,
    pub prompt_hash: String,
    pub response_quality: f64,
    pub latency_score: f64,
    pub cost_efficiency: f64,
    pub user_feedback: Option<String>,
}

pub struct Database {
    db: SharedDrizzleDb,
    schema: SentinelSchema,
}

impl Clone for Database {
    fn clone(&self) -> Self {
        Self {
            db: self.db.clone(),
            schema: SentinelSchema::new(),
        }
    }
}

impl std::fmt::Debug for Database {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Database").finish()
    }
}

#[derive(SQLiteFromRow, Default)]
struct StatsScanRow(f64, i64, bool, bool, f64, String);

#[derive(SQLiteFromRow, Default)]
struct MonthlyScanRow(String, f64, f64);

#[derive(SQLiteFromRow, Default)]
struct ModelScanRow(String, String, i64, f64, f64);

impl Database {
    pub async fn new(path: &PathBuf) -> anyhow::Result<Self> {
        let conn = rusqlite::Connection::open(path)?;
        let (db, schema) = drizzle::sqlite::rusqlite::Drizzle::new(conn, SentinelSchema::new());
        let _ = db.create();

        Ok(Self {
            db: Arc::new(Mutex::new(db)),
            schema,
        })
    }

    pub fn get_connection(&self) -> SharedDrizzleDb {
        self.db.clone()
    }

    pub async fn log_request(&self, log: &RequestLog) -> anyhow::Result<()> {
        let db = self
            .db
            .lock()
            .map_err(|_| anyhow::anyhow!("database mutex poisoned"))?;

        let base = InsertLogsTable::new(
            log.id.to_string(),
            log.timestamp.to_rfc3339(),
            log.provider.clone(),
            log.model.clone(),
            log.input_tokens as i64,
            log.output_tokens as i64,
            log.latency_ms as i64,
            log.cost_usd,
            log.cache_hit,
            log.status.clone(),
        )
        .with_cost_saved(log.cost_saved)
        .with_quality_score(log.quality_score)
        .with_pii_redacted(log.pii_redacted);

        if let Some(message) = &log.error_message {
            db.insert(self.schema.logs)
                .values([base.with_error_message(message.clone())])
                .execute()?;
        } else {
            db.insert(self.schema.logs).values([base]).execute()?;
        }

        Ok(())
    }

    pub async fn get_recent_logs(&self, limit: usize) -> anyhow::Result<Vec<RequestLog>> {
        let db = self
            .db
            .lock()
            .map_err(|_| anyhow::anyhow!("database mutex poisoned"))?;

        let rows: Vec<SelectLogsTable> = db
            .select(())
            .from(self.schema.logs)
            .order_by([OrderBy::desc(self.schema.logs.timestamp)])
            .limit(limit)
            .all()?;

        let mut out = Vec::with_capacity(rows.len());
        for row in rows {
            out.push(RequestLog {
                id: Uuid::parse_str(&row.id)?,
                timestamp: DateTime::parse_from_rfc3339(&row.timestamp)?.with_timezone(&Utc),
                provider: row.provider,
                model: row.model,
                input_tokens: row.input_tokens as u32,
                output_tokens: row.output_tokens as u32,
                latency_ms: row.latency_ms as u32,
                cost_usd: row.cost_usd,
                cost_saved: row.cost_saved,
                quality_score: row.quality_score,
                cache_hit: row.cache_hit,
                pii_redacted: row.pii_redacted,
                status: row.status,
                error_message: row.error_message,
            });
        }

        Ok(out)
    }

    pub async fn get_stats(&self) -> anyhow::Result<serde_json::Value> {
        let db = self
            .db
            .lock()
            .map_err(|_| anyhow::anyhow!("database mutex poisoned"))?;

        let rows: Vec<StatsScanRow> = db
            .select((
                self.schema.logs.cost_usd,
                self.schema.logs.latency_ms,
                self.schema.logs.cache_hit,
                self.schema.logs.pii_redacted,
                self.schema.logs.cost_saved,
                self.schema.logs.timestamp,
            ))
            .from(self.schema.logs)
            .all()?;

        let mut total_requests: u32 = 0;
        let mut total_cost = 0.0;
        let mut total_latency = 0.0;
        let mut cache_hits: u32 = 0;
        let mut pii_redacted: u32 = 0;
        let mut total_saved = 0.0;
        let mut monthly_saved = 0.0;
        let current_month = Utc::now().format("%Y-%m").to_string();

        for row in rows {
            total_requests += 1;
            total_cost += row.0;
            total_latency += row.1 as f64;
            if row.2 {
                cache_hits += 1;
            }
            if row.3 {
                pii_redacted += 1;
            }
            total_saved += row.4;
            if row.5.starts_with(&current_month) {
                monthly_saved += row.4;
            }
        }

        let avg_latency = if total_requests > 0 {
            total_latency / total_requests as f64
        } else {
            0.0
        };

        Ok(serde_json::json!({
            "total_requests": total_requests,
            "total_cost": total_cost,
            "avg_latency": avg_latency,
            "cache_hits": cache_hits,
            "pii_redacted": pii_redacted,
            "total_saved": total_saved,
            "monthly_saved": monthly_saved,
        }))
    }

    pub async fn log_cost_optimization(&self, opt: &CostOptimization) -> anyhow::Result<()> {
        let db = self
            .db
            .lock()
            .map_err(|_| anyhow::anyhow!("database mutex poisoned"))?;

        db.insert(self.schema.cost_optimizations)
            .values([InsertCostOptimizationsTable::new(
                opt.id.to_string(),
                opt.timestamp.to_rfc3339(),
                opt.endpoint_pattern.clone(),
                opt.original_model.clone(),
                opt.suggested_model.clone(),
                opt.potential_savings_percent,
                opt.confidence_score,
                opt.reason.clone(),
                opt.status.clone(),
            )])
            .execute()?;

        Ok(())
    }

    pub async fn get_pending_optimizations(&self) -> anyhow::Result<Vec<CostOptimization>> {
        let db = self
            .db
            .lock()
            .map_err(|_| anyhow::anyhow!("database mutex poisoned"))?;

        let rows: Vec<SelectCostOptimizationsTable> = db
            .select(())
            .from(self.schema.cost_optimizations)
            .r#where(eq(self.schema.cost_optimizations.status, "pending"))
            .order_by([OrderBy::desc(
                self.schema.cost_optimizations.confidence_score,
            )])
            .limit(10)
            .all()?;

        let mut out = Vec::with_capacity(rows.len());
        for row in rows {
            out.push(CostOptimization {
                id: Uuid::parse_str(&row.id)?,
                timestamp: DateTime::parse_from_rfc3339(&row.timestamp)?.with_timezone(&Utc),
                endpoint_pattern: row.endpoint_pattern,
                original_model: row.original_model,
                suggested_model: row.suggested_model,
                potential_savings_percent: row.potential_savings_percent,
                confidence_score: row.confidence_score,
                reason: row.reason,
                status: row.status,
            });
        }

        Ok(out)
    }

    pub async fn log_quality_metric(&self, metric: &QualityMetric) -> anyhow::Result<()> {
        let db = self
            .db
            .lock()
            .map_err(|_| anyhow::anyhow!("database mutex poisoned"))?;

        let base = InsertQualityMetricsTable::new(
            metric.id.to_string(),
            metric.timestamp.to_rfc3339(),
            metric.model.clone(),
            metric.prompt_hash.clone(),
            metric.response_quality,
            metric.latency_score,
            metric.cost_efficiency,
        );

        if let Some(feedback) = &metric.user_feedback {
            db.insert(self.schema.quality_metrics)
                .values([base.with_user_feedback(feedback.clone())])
                .execute()?;
        } else {
            db.insert(self.schema.quality_metrics)
                .values([base])
                .execute()?;
        }

        Ok(())
    }

    pub async fn get_monthly_savings(&self, months_back: i32) -> anyhow::Result<serde_json::Value> {
        let db = self
            .db
            .lock()
            .map_err(|_| anyhow::anyhow!("database mutex poisoned"))?;

        let rows: Vec<MonthlyScanRow> = db
            .select((
                self.schema.logs.timestamp,
                self.schema.logs.cost_saved,
                self.schema.logs.cost_usd,
            ))
            .from(self.schema.logs)
            .all()?;

        let cutoff = Utc::now() - chrono::Duration::days((months_back as i64) * 30);
        let mut monthly_map: HashMap<String, (f64, u32, f64)> = HashMap::new();
        let mut total_saved = 0.0;

        for row in rows {
            let ts = match DateTime::parse_from_rfc3339(&row.0) {
                Ok(v) => v.with_timezone(&Utc),
                Err(_) => continue,
            };
            if ts < cutoff {
                continue;
            }

            let month = ts.format("%Y-%m").to_string();
            let entry = monthly_map.entry(month).or_insert((0.0, 0, 0.0));
            entry.0 += row.1;
            entry.1 += 1;
            entry.2 += row.2;
            total_saved += row.1;
        }

        let mut months: Vec<_> = monthly_map.into_iter().collect();
        months.sort_by(|a, b| b.0.cmp(&a.0));

        let monthly_data: Vec<serde_json::Value> = months
            .into_iter()
            .map(|(month, (saved, requests, spent))| {
                serde_json::json!({
                    "month": month,
                    "saved": saved,
                    "requests": requests,
                    "spent": spent
                })
            })
            .collect();

        Ok(serde_json::json!({
            "total_saved": total_saved,
            "monthly_breakdown": monthly_data
        }))
    }

    pub async fn get_model_performance_stats(&self) -> anyhow::Result<serde_json::Value> {
        let db = self
            .db
            .lock()
            .map_err(|_| anyhow::anyhow!("database mutex poisoned"))?;

        let rows: Vec<ModelScanRow> = db
            .select((
                self.schema.logs.model,
                self.schema.logs.timestamp,
                self.schema.logs.latency_ms,
                self.schema.logs.cost_usd,
                self.schema.logs.quality_score,
            ))
            .from(self.schema.logs)
            .all()?;

        let cutoff = Utc::now() - chrono::Duration::days(30);
        let mut by_model: HashMap<String, (u32, f64, f64, f64)> = HashMap::new();

        for row in rows {
            let ts = match DateTime::parse_from_rfc3339(&row.1) {
                Ok(v) => v.with_timezone(&Utc),
                Err(_) => continue,
            };
            if ts < cutoff {
                continue;
            }

            let entry = by_model.entry(row.0).or_insert((0, 0.0, 0.0, 0.0));
            entry.0 += 1;
            entry.1 += row.2 as f64;
            entry.2 += row.3;
            entry.3 += row.4;
        }

        let mut model_stats = Vec::new();
        for (model, (requests, latency_sum, cost_sum, quality_sum)) in by_model {
            let avg_cost = if requests > 0 {
                cost_sum / requests as f64
            } else {
                0.0
            };
            let avg_quality = if requests > 0 {
                quality_sum / requests as f64
            } else {
                0.5
            };
            let avg_latency = if requests > 0 {
                latency_sum / requests as f64
            } else {
                0.0
            };
            let quality_per_dollar = if avg_cost > 0.0 {
                avg_quality / avg_cost
            } else {
                0.0
            };

            model_stats.push(serde_json::json!({
                "model": model,
                "requests": requests,
                "avg_latency": avg_latency,
                "avg_cost": avg_cost,
                "avg_quality": avg_quality,
                "total_cost": cost_sum,
                "quality_per_dollar": quality_per_dollar
            }));
        }
        model_stats.sort_by(|a, b| {
            b["requests"]
                .as_u64()
                .unwrap_or(0)
                .cmp(&a["requests"].as_u64().unwrap_or(0))
        });

        Ok(serde_json::json!({
            "models": model_stats
        }))
    }
}
