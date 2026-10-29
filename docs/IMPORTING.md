# Importing datasets (§ 14 Datasets)

Studio's built-in downloads cover a few datasets; any other can be imported by hand.

1. **Where from.** A Hugging Face dataset (its link or `owner/name`, at a revision: `main`, a
   branch, a tag or a commit), an https link to a file, or files on this device (also inside a
   ZIP). Gated and private datasets use your Hugging Face account (§ 4).
2. **Files.** Every file of the revision is listed with its split (train, validation, test, from
   its path), format and size, and so is Hugging Face's own Parquet conversion when it has one.
   Readable: Parquet, JSON, JSON lines, CSV, TSV and plain text, each also gzip-compressed (`.gz`),
   and ZIPs of those. **Python loading scripts are never run.** A dataset built by one is noted;
   the script is read as text for the links to its data files, and those with https links are
   offered (an http link is named but not downloaded). Tick files of one split: train,
   validation and test are imported as separate datasets, never mixed.
3. **Columns and mapping.** The files' columns (kind and an example) and a suggested mapping:
   - text to read: one or more text columns (§ 2 Reading; stories, facts or worked problems);
   - conversations (§ 4): a list of messages with a role and a text (any field names), a prompt
     and a response, a list of turns, or one record per turn grouped by a conversation column.
     Roles can be named or left out (a system prompt, for instance); speakers that are not
     named get names. A scene column becomes the conversation's `Situation:`.
   **Preview** converts the first 2,000 records and shows the first documents and what was left
   out, and the roles found.
4. **Import** converts every record, up to the size chosen. Left out, and counted: records that
   cannot be read, a missing field, a role not mapped (once roles are named), an empty message,
   a malformed record, the same speaker twice in a row, more than two speakers, over 20,000
   characters, duplicates, and (one record per turn) a conversation whose records are not
   together. A conversation dataset can have the quality filter's switches (§ 4) apply to it at
   practice (on for Hugging Face and links, as for downloads) or not (as for your own
   conversations).

Kept with the dataset: the source, revision, split, every original file's name, SHA-256 and size,
the mapping, the import's counts and whether the filter applies.

Limits: a file up to 4 GB (a JSON file holding one array up to 512 MB: JSON lines can be any
size, as they are read record by record); a ZIP with at most 2,000 entries and 4 GB unpacked;
entries with unsafe paths (`..`, absolute) refuse the whole ZIP, and an entry unpacking to more
than 200 times its packed size, or more than it declares, is refused. Downloads are cached while
you work on an import (resumable) and removed after it.
