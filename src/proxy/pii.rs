use regex::Regex;
use std::sync::OnceLock;

pub struct PiiRedactor {
    patterns: Vec<(&'static str, Regex)>,
}

impl Default for PiiRedactor {
    fn default() -> Self {
        Self::new()
    }
}

impl PiiRedactor {
    pub fn new() -> Self {
        let patterns = vec![
            (
                "EMAIL",
                Regex::new(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}").unwrap(),
            ),
            (
                "PHONE",
                Regex::new(r"\b(?:\+?1[-. ]?)?\(?([0-9]{3})\)?[-. ]?([0-9]{3})[-. ]?([0-9]{4})\b")
                    .unwrap(),
            ),
            ("SSN", Regex::new(r"\b\d{3}-\d{2}-\d{4}\b").unwrap()),
            (
                "CREDIT_CARD",
                Regex::new(r"\b(?:\d[ -]*?){13,16}\b").unwrap(),
            ),
            (
                "API_KEY",
                Regex::new(r"(sk-[a-zA-Z0-9]{20,}|Bearer\s+[a-zA-Z0-9._-]+)").unwrap(),
            ),
        ];
        Self { patterns }
    }

    pub fn redact(&self, text: &str) -> (String, bool) {
        let mut redacted = text.to_string();
        let mut changed = false;
        for (label, regex) in &self.patterns {
            let result = regex
                .replace_all(&redacted, &format!("[{}_REDACTED]", label))
                .to_string();
            if result != redacted {
                redacted = result;
                changed = true;
            }
        }
        (redacted, changed)
    }
}

pub static REDACTOR: OnceLock<PiiRedactor> = OnceLock::new();

pub fn get_redactor() -> &'static PiiRedactor {
    REDACTOR.get_or_init(PiiRedactor::new)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pii_redaction() {
        let redactor = PiiRedactor::new();
        let input =
            "Contact me at john.doe@example.com or call 555-123-4567. My SSN is 123-45-6789.";
        let (redacted, changed) = redactor.redact(input);
        assert!(changed);
        assert!(redacted.contains("[EMAIL_REDACTED]"));
        assert!(redacted.contains("[PHONE_REDACTED]"));
        assert!(redacted.contains("[SSN_REDACTED]"));
        assert!(!redacted.contains("john.doe@example.com"));
        assert!(!redacted.contains("555-123-4567"));
        assert!(!redacted.contains("123-45-6789"));
    }
}
