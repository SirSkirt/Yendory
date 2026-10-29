# Hugging Face account and downloads

## Connecting

Studio uses a Hugging Face **access token** (Hugging Face does not offer a sign-in flow Studio
could use without registering an OAuth app of its own; a token is the supported way). A **read**
token is enough: Hugging Face → Settings → Access Tokens → New token → Read.

On § 4 Conversation, paste it and press **Save token**. Studio checks it with Hugging Face
(`/api/whoami-v2`) and shows the account it belongs to, the token's name and its kind. A token that
can write is accepted but Studio suggests a read token. **Check** asks again (an expired or revoked
token shows as not accepted); **Disconnect** removes it.

## Where the token is kept

| System | Kept in |
|---|---|
| Windows | Windows Credential Manager |
| Linux | your desktop's keyring (the Secret Service) |
| macOS | the Keychain |
| Android, or a Linux session without a keyring | `hf-token.secret` in Studio's data folder, readable only by your user account (the page says so) |

It is never kept in the settings file. A token saved by an earlier version (in `settings.json`) is
moved at start-up and removed from the settings file and from every copy of it (the backup, the
copy kept when the settings were upgraded). It is sent only to huggingface.co (never on a redirect
to another host), never back to the page, and never written to logs, history or exports.

## Gated datasets

A token is not access to every gated dataset. LMSYS-Chat-1M, for instance, asks you to accept its
conditions on its page while signed in. Without that, Hugging Face answers 401 or 403 and Studio
says what to do; these are not retried.

## Downloads

Datasets are read from the Parquet files Hugging Face publishes for them (about 100 times faster
than the dataset viewer's pages of 100 rows; the pages remain the fallback). Files are downloaded
one at a time, continued after a break with a range request, read and deleted. Conversation
downloads keep every usable conversation; the quality filter is chosen on § 4 and applied when
practice starts.

## The download queue

Downloads run one at a time, in order, and the queue is saved (`download_queue.json`) after every
change, so it survives closing Studio, a crash or an update. When Studio starts, the queue comes
back **paused**, with a download that was running first in line, unless "Resume the download queue
when Studio starts" (§ 1 Models) is on. In the queue: **Pause / Resume** (the running download
stops and waits first in line; a Parquet file being downloaded continues from where it stopped,
though files of that dataset already read are read again), **↑ / ↓** to reorder, **Remove** (takes
it out of the line; nothing downloaded is deleted), **Retry** for a failed or cancelled one, and
**Clear finished**. Failed downloads keep their reason; they are not retried by themselves.
