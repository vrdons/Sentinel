use libsql::Connection;
use uuid::Uuid;
use chrono::{DateTime, Utc};
use serde::{Serialize, Deserialize};
use std::path::PathBuf;
use std::sync::Arc;

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

#[derive(Clone)]
pub struct Database {
    conn: Connection,
}

impl std::fmt::Debug for Database {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Database").finish()
    }
}

impl Database {
    pub async fn new(path: &PathBuf) -> anyhow::Result<Self> {
        let db = libsql::Builder::new_local(path).build().await?;
        let conn = db.connect()?;

        conn.execute(
            "CREATE TABLE IF NOT EXISTS logs (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                provider TEXT,
                model TEXT,
                input_tokens INTEGER,
                output_tokens INTEGER,
                latency_ms INTEGER,
                cost_usd REAL,
                cost_saved REAL DEFAULT 0.0,
                quality_score REAL DEFAULT 0.5,
                cache_hit INTEGER,
                pii_redacted INTEGER DEFAULT 0,
                status TEXT,
                error_message TEXT
            )",
            (),
        ).await?;

        conn.execute(
            "CREATE TABLE IF NOT EXISTS cost_optimizations (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                endpoint_pattern TEXT,
                original_model TEXT,
                suggested_model TEXT,
                potential_savings_percent REAL,
                confidence_score REAL,
                reason TEXT,
                status TEXT
            )",
            (),
        ).await?;

        conn.execute(
            "CREATE TABLE IF NOT EXISTS quality_metrics (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                model TEXT,
                prompt_hash TEXT,
                response_quality REAL,
                latency_score REAL,
                cost_efficiency REAL,
                user_feedback TEXT
            )",
            (),
        ).await?;

        // Migrations
        let _ = conn.execute("ALTER TABLE logs ADD COLUMN pii_redacted INTEGER DEFAULT 0", ()).await;
        let _ = conn.execute("ALTER TABLE logs ADD COLUMN cost_saved REAL DEFAULT 0.0", ()).await;
        let _ = conn.execute("ALTER TABLE logs ADD COLUMN quality_score REAL DEFAULT 0.5", ()).await;

        Ok(Self { conn })
    }

    pub fn get_connection(&self) -> Arc<Connection> {
        Arc::new(self.conn.clone())
    }

    pub async fn log_request(&self, log: &RequestLog) -> anyhow::Result<()> {
        self.conn.execute(
            "INSERT INTO logs (
                id, timestamp, provider, model, input_tokens, output_tokens,
                latency_ms, cost_usd, cost_saved, quality_score, cache_hit, pii_redacted, status, error_message
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14)",
            libsql::params![
                log.id.to_string(),
                log.timestamp.to_rfc3339(),
                log.provider.clone(),
                log.model.clone(),
                log.input_tokens,
                log.output_tokens,
                log.latency_ms,
                log.cost_usd,
                log.cost_saved,
                log.quality_score,
                if log.cache_hit { 1 } else { 0 },
                if log.pii_redacted { 1 } else { 0 },
                log.status.clone(),
                log.error_message.clone(),
            ],
        ).await?;

        Ok(())
    }

    pub async fn get_recent_logs(&self, limit: usize) -> anyhow::Result<Vec<RequestLog>> {
        let mut rows = self.conn.query(
            "SELECT id, timestamp, provider, model, input_tokens, output_tokens, latency_ms, cost_usd, cost_saved, quality_score, cache_hit, pii_redacted, status, error_message
             FROM logs ORDER BY timestamp DESC LIMIT ?1",
            libsql::params![limit as u32],
        ).await?;

        let mut logs = Vec::new();
        while let Some(row) = rows.next().await? {
            logs.push(RequestLog {
                id: Uuid::parse_str(&row.get::<String>(0)?)?,
                timestamp: DateTime::parse_from_rfc3339(&row.get::<String>(1)?)?.with_timezone(&Utc),
                provider: row.get::<String>(2)?,
                model: row.get::<String>(3)?,
                input_tokens: row.get::<u32>(4)?,
                output_tokens: row.get::<u32>(5)?,
                latency_ms: row.get::<u32>(6)?,
                cost_usd: row.get::<f64>(7)?,
                cost_saved: row.get::<f64>(8).unwrap_or(0.0),
                quality_score: row.get::<f64>(9).unwrap_or(0.5),
                cache_hit: row.get::<i32>(10)? != 0,
                pii_redacted: row.get::<i32>(11)? != 0,
                status: row.get::<String>(12)?,
                error_message: row.get::<Option<String>>(13)?,
            });
        }
        Ok(logs)
    }

    pub async fn get_stats(&self) -> anyhow::Result<serde_json::Value> {
        let mut rows = self.conn.query(
            "SELECT
                COUNT(*),
                SUM(cost_usd),
                AVG(latency_ms),
                SUM(cache_hit),
                SUM(pii_redacted),
                SUM(COALESCE(cost_saved, 0.0))
             FROM logs",
            (),
        ).await?;

        if let Some(row) = rows.next().await? {
            let total_cost = row.get::<f64>(1).unwrap_or(0.0);
            let total_saved = row.get::<f64>(5).unwrap_or(0.0);

            // Get current month savings
            let mut monthly_rows = self.conn.query(
                "SELECT SUM(COALESCE(cost_saved, 0.0)) FROM logs
                 WHERE strftime('%Y-%m', timestamp) = strftime('%Y-%m', 'now')",
                (),
            ).await?;

            let monthly_saved = if let Some(m_row) = monthly_rows.next().await? {
                m_row.get::<f64>(0).unwrap_or(0.0)
            } else {
                0.0
            };

            Ok(serde_json::json!({
                "total_requests": row.get::<u32>(0)?,
                "total_cost": total_cost,
                "avg_latency": row.get::<f64>(2).unwrap_or(0.0),
                "cache_hits": row.get::<u32>(3).unwrap_or(0),
                "pii_redacted": row.get::<u32>(4).unwrap_or(0),
                "total_saved": total_saved,
                "monthly_saved": monthly_saved,
            }))
        } else {
            Ok(serde_json::json!({}))
        }
    }

    pub async fn log_cost_optimization(&self, opt: &CostOptimization) -> anyhow::Result<()> {
        self.conn.execute(
            "INSERT INTO cost_optimizations (
                id, timestamp, endpoint_pattern, original_model, suggested_model,
                potential_savings_percent, confidence_score, reason, status
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            libsql::params![
                opt.id.to_string(),
                opt.timestamp.to_rfc3339(),
                opt.endpoint_pattern.clone(),
                opt.original_model.clone(),
                opt.suggested_model.clone(),
                opt.potential_savings_percent,
                opt.confidence_score,
                opt.reason.clone(),
                opt.status.clone(),
            ],
        ).await?;

        Ok(())
    }

    pub async fn get_pending_optimizations(&self) -> anyhow::Result<Vec<CostOptimization>> {
        let mut rows = self.conn.query(
            "SELECT id, timestamp, endpoint_pattern, original_model, suggested_model,
                    potential_savings_percent, confidence_score, reason, status
             FROM cost_optimizations WHERE status = 'pending' ORDER BY confidence_score DESC LIMIT 10",
            (),
        ).await?;

        let mut opts = Vec::new();
        while let Some(row) = rows.next().await? {
            opts.push(CostOptimization {
                id: Uuid::parse_str(&row.get::<String>(0)?)?,
                timestamp: DateTime::parse_from_rfc3339(&row.get::<String>(1)?)?.with_timezone(&Utc),
                endpoint_pattern: row.get::<String>(2)?,
                original_model: row.get::<String>(3)?,
                suggested_model: row.get::<String>(4)?,
                potential_savings_percent: row.get::<f64>(5)?,
                confidence_score: row.get::<f64>(6)?,
                reason: row.get::<String>(7)?,
                status: row.get::<String>(8)?,
            });
        }
        Ok(opts)
    }

    pub async fn log_quality_metric(&self, metric: &QualityMetric) -> anyhow::Result<()> {
        self.conn.execute(
            "INSERT INTO quality_metrics (
                id, timestamp, model, prompt_hash, response_quality,
                latency_score, cost_efficiency, user_feedback
            ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            libsql::params![
                metric.id.to_string(),
                metric.timestamp.to_rfc3339(),
                metric.model.clone(),
                metric.prompt_hash.clone(),
                metric.response_quality,
                metric.latency_score,
                metric.cost_efficiency,
                metric.user_feedback.clone(),
            ],
        ).await?;

        Ok(())
    }

    pub async fn get_monthly_savings(&self, months_back: i32) -> anyhow::Result<serde_json::Value> {
        let mut rows = self.conn.query(
            "SELECT 
                strftime('%Y-%m', timestamp) as month,
                SUM(COALESCE(cost_saved, 0.0)) as monthly_saved,
                COUNT(*) as requests,
                SUM(cost_usd) as spent
             FROM logs 
             WHERE datetime(timestamp) >= datetime('now', '-' || ?1 || ' months')
             GROUP BY strftime('%Y-%m', timestamp)
             ORDER BY month DESC",
            libsql::params![months_back],
        ).await?;

        let mut monthly_data = Vec::new();
        let mut total_saved = 0.0;
        
        while let Some(row) = rows.next().await? {
            let saved = row.get::<f64>(1).unwrap_or(0.0);
            total_saved += saved;
            
            monthly_data.push(serde_json::json!({
                "month": row.get::<String>(0)?,
                "saved": saved,
                "requests": row.get::<u32>(2)?,
                "spent": row.get::<f64>(3).unwrap_or(0.0)
            }));
        }

        Ok(serde_json::json!({
            "total_saved": total_saved,
            "monthly_breakdown": monthly_data
        }))
    }

    pub async fn get_model_performance_stats(&self) -> anyhow::Result<serde_json::Value> {
        let mut rows = self.conn.query(
            "SELECT 
                model,
                COUNT(*) as requests,
                AVG(latency_ms) as avg_latency,
                AVG(cost_usd) as avg_cost,
                AVG(COALESCE(quality_score, 0.5)) as avg_quality,
                SUM(cost_usd) as total_cost
             FROM logs 
             WHERE datetime(timestamp) >= datetime('now', '-30 days')
             GROUP BY model
             ORDER BY requests DESC",
            (),
        ).await?;

        let mut model_stats = Vec::new();
        
        while let Some(row) = rows.next().await? {
            let avg_cost = row.get::<f64>(3).unwrap_or(0.0);
            let avg_quality = row.get::<f64>(4).unwrap_or(0.5);
            let quality_per_dollar = if avg_cost > 0.0 { avg_quality / avg_cost } else { 0.0 };
            
            model_stats.push(serde_json::json!({
                "model": row.get::<String>(0)?,
                "requests": row.get::<u32>(1)?,
                "avg_latency": row.get::<f64>(2).unwrap_or(0.0),
                "avg_cost": avg_cost,
                "avg_quality": avg_quality,
                "total_cost": row.get::<f64>(5).unwrap_or(0.0),
                "quality_per_dollar": quality_per_dollar
            }));
        }

        Ok(serde_json::json!({
            "models": model_stats
        }))
    }
}
