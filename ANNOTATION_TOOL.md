# QV-M2 video annotation demo

This release adds a lightweight browser-based annotation tool and a small
English JSONL sample for trying the workflow without publishing videos,
features, checkpoints, or a working annotation database.

## Included files

- `tools/video_annotation_app.py`: standard-library-only annotation server.
- `tools/video_annotation_app_README.md`: detailed usage and workflow guide.
- `data/samples/qv_m2_test_en_100.jsonl`: 100 unchanged English records from
  the local QV-M2 test annotations.
- `data/samples/README.md` and `data/samples/LICENSE`: provenance and data
  licensing information.

## Start locally

From the repository root:

```powershell
python tools\video_annotation_app.py --host 127.0.0.1 --port 7860
```

Open <http://127.0.0.1:7860> and import
`data/samples/qv_m2_test_en_100.jsonl` from the reviewer interface.

To associate records with local videos whose filenames match `vid.mp4`:

```powershell
python tools\video_annotation_app.py `
  --host 127.0.0.1 `
  --port 7860 `
  --video_root D:\path\to\videos `
  --import_jsonl test:data\samples\qv_m2_test_en_100.jsonl
```

For LAN use, set `--host 0.0.0.0`, restrict the firewall rule to the private
local subnet, and do not expose this development server directly to the
public Internet.

Runtime state is written to `annotation_workspace/`. Do not commit that
directory because it can contain user accounts, task assignments, uploads,
and in-progress annotations.

## Data availability

The sample contains annotation metadata only. It does not include video
files, extracted features, subtitles, or model checkpoints. See
`data/samples/README.md` for provenance and license terms.

