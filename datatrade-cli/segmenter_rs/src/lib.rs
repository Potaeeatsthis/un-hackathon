use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use regex::Regex;
use std::sync::OnceLock;

// Compiled once — same patterns as config.py SEGMENT_PATTERNS
static PATTERNS: OnceLock<Vec<Regex>> = OnceLock::new();

fn get_patterns() -> &'static Vec<Regex> {
    PATTERNS.get_or_init(|| {
        vec![
            Regex::new(r"(?i)^(section|article|มาตรา|pasal|điều|clause)\s*[\d]+").unwrap(),
            Regex::new(r"(?i)^(\d+[.\)])\s+\w").unwrap(),
        ]
    })
}

const MIN_SEGMENT_CHARS: usize = 80;

fn match_header(line: &str) -> Option<String> {
    let trimmed = line.trim();
    for pat in get_patterns() {
        if pat.is_match(trimmed) {
            // Sanitise first 40 chars into a slug (spaces → dashes)
            let slug: String = trimmed
                .chars()
                .take(40)
                .collect::<String>()
                .split_whitespace()
                .collect::<Vec<_>>()
                .join("-");
            return Some(slug);
        }
    }
    None
}

/// segment(pages, doc_name, country, min_chars) -> list[dict]
///
/// pages: list of {"page": int, "text": str}
/// Returns list of {"section_id": str, "page": int, "text": str,
///                  "doc_name": str, "country": str}
#[pyfunction]
#[pyo3(signature = (pages, doc_name, country="XX", min_chars=80))]
fn segment(
    py: Python<'_>,
    pages: &Bound<'_, PyList>,
    doc_name: &str,
    country: &str,
    min_chars: usize,
) -> PyResult<Vec<PyObject>> {
    let min_chars = if min_chars == 80 { MIN_SEGMENT_CHARS } else { min_chars };

    // Build flat line list: (line_text, page_number)
    let mut lines: Vec<(String, i64)> = Vec::new();
    for item in pages.iter() {
        let page_dict = item.downcast::<PyDict>()?;
        let page_num: i64 = page_dict
            .get_item("page")?
            .map(|v| v.extract::<i64>().unwrap_or(1))
            .unwrap_or(1);
        let text: String = page_dict
            .get_item("text")?
            .map(|v| v.extract::<String>().unwrap_or_default())
            .unwrap_or_default();
        for line in text.lines() {
            lines.push((line.to_string(), page_num));
        }
    }

    let mut segments: Vec<PyObject> = Vec::new();
    let mut current_lines: Vec<String> = Vec::new();
    let mut current_page: i64 = lines.first().map(|(_, p)| *p).unwrap_or(1);
    let mut current_id = format!("{}-PREAMBLE", country);
    let mut seq: usize = 0;

    for (line_text, page_num) in &lines {
        if let Some(header) = match_header(line_text) {
            // Flush previous segment
            let chunk = current_lines.join("\n");
            let chunk = chunk.trim();
            if chunk.chars().count() >= min_chars {
                let d = PyDict::new(py);
                d.set_item("section_id", &current_id)?;
                d.set_item("page", current_page)?;
                d.set_item("text", chunk)?;
                d.set_item("doc_name", doc_name)?;
                d.set_item("country", country)?;
                segments.push(d.into());
            }
            seq += 1;
            current_id = format!("{}-{}-{}", country, header, seq);
            current_lines = vec![line_text.clone()];
            current_page = *page_num;
        } else {
            current_lines.push(line_text.clone());
        }
    }

    // Flush last segment
    let chunk = current_lines.join("\n");
    let chunk = chunk.trim();
    if chunk.chars().count() >= min_chars {
        let d = PyDict::new(py);
        d.set_item("section_id", &current_id)?;
        d.set_item("page", current_page)?;
        d.set_item("text", chunk)?;
        d.set_item("doc_name", doc_name)?;
        d.set_item("country", country)?;
        segments.push(d.into());
    }

    Ok(segments)
}

#[pymodule]
fn segmenter_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(segment, m)?)?;
    Ok(())
}
