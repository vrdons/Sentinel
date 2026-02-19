use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "sentinel")]
#[command(about = "A fast, reliable LLM gateway", long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Start the sentinel proxy and dashboard
    Start {
        /// Start only the proxy
        #[arg(long)]
        proxy_only: bool,

        /// Start only the dashboard
        #[arg(long)]
        dashboard_only: bool,
    },
    /// Show recent logs
    Logs {
        /// Number of lines to show
        #[arg(long)]
        tail: Option<usize>,

        /// Follow the logs in real-time
        #[arg(long, short)]
        follow: bool,
    },
    /// Show current configuration
    Config,
}
