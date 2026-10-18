//! Study Mode memory (training memory). Filled in by `learn::study`.

/// Starts a memory line in the model's context: a remembered statement Study Mode retrieved.
/// Like the Nexus markers it never occurs in outside text (lesson files and user messages have
/// it removed), so only retrieval can put a memory in front of the model.
pub const MEMORY: char = '\x03';
