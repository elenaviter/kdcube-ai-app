
## Stack
- **Storage backend**: s3 / local filesystem.
  Configured in KnowledgeBase object creation. 
  ```python
  KnowledgeBase("running-shoes", f"file://{STORAGE_PATH}/running-shoes/knowledge_base")
  ```
  - **Client-Server live notifications**: Socket IO
    - Used to connect the web clients and the backend for internal events exchange. 
    - Works with both web socket and long polling transport.
    - Client subscribe to the events of interest and can provide the room id to listen.
    - Room id is by default unique to the client session, but there might be "shared rooms" for some shared events in the system. 

  - **Messaging Service**: Redis
    - Allow notifications at scale by conveying the targeted (or broadcast) messages from their originating processes to the connected Socket IO clients.
    - Used as pub/sub. Async working processes can run at scale and pub the events to dedicated Redis channels. This allows to route messages to the clients that are subscribed for updates (room-specific or broadcast).  

## Current code organization
### Backend:
- [KnowledgeBaseStorage](storage.py)
- [Supported KB Artifacts Storage Backends](../../storage/storage.py)
- [Data Elements and Data Source models](../../data_model/datasource.py)
- [Tools: Data Parsers](../../tools/parser.py)
- [Tools: Data Extraction](../../tools/extract.py)
- [Tools: Data Fetch](../../tools/fetch.py)
- [KnowledgeBase](core.py)
- [KnowledgeBase HTTP and Socket IO Server](api.py)
- [KnowledgeBase Celery Server](orchestrator_celery.py)
- [Dataproc: Pipeline and base definition of the module](modules/base.py)
- [Dataproc: Extraction Module](modules/extraction.py)
- [Dataproc: Segmentatation Module](modules/segmentation.py)
- [Dataproc: Metadata Module](modules/metadata.py)
- [Search: Search Engine](search.py)
## Current data organization
```bash
<project_name>
└── knowledge_base
    ├── data
    │   ├── embedding
    │   ├── extraction
    │   │   ├── file|products.pdf
    │   │   │   └── versions
    │   │   │       └── 1
    │   │   │           ├── extraction.json
    │   │   │           ├── extraction_0.md
    │   │   │           └── extraction_0_metadata_0.json
    │   │   ├── url|arxiv.org_pdf_2505.00661
    │   │   │   └── versions
    │   │   │       └── 1
    │   │   │           ├── extraction.json
    │   │   │           ├── extraction_0.md
    │   │   │           ├── extraction_0_image_0.jpeg
    │   │   │           └── extraction_0_metadata_0.json
    │   │   ├── url|ridgerunai.medium.com_introducing-voice-agent-real-time-voice-assistant-for-language-models-b70fa6f2d593
    │   │   │   └── versions
    │   │   │       └── 1
    │   │   │           ├── extraction.json
    │   │   │           └── extraction_0.md
    │   ├── metadata
    │   │   ├── file|products.pdf
    │   │   │   └── versions
    │   │   │       └── 1
    │   │   │           ├── curriculum
    │   │   │           │   ├── metadata.json
    │   │   │           │   └── segment_metadata_0.json
    │   │   │           ├── metadata.json
    │   │   │           └── retrieval
    │   │   │               ├── metadata.json
    │   │   │               └── segment_metadata_0.json
    │   │   ├── url|arxiv.org_pdf_2505.00661
    │   │   │   └── versions
    │   │   │       └── 1
    │   │   │           ├── curriculum
    │   │   │           │   ├── metadata.json
    │   │   │           │   ├── segment_metadata_N.json
    │   │   │           ├── metadata.json
    │   │   │           └── retrieval
    │   │   │               ├── metadata.json
    │   │   │               ├── segment_metadata_M.json
    │   │   ├── url|ridgerunai.medium.com_introducing-voice-agent-real-time-voice-assistant-for-language-models-b70fa6f2d593
    │   │   │   └── versions
    │   │   │       └── 1
    │   │   │           ├── curriculum
    │   │   │           │   ├── metadata.json
    │   │   │           │   ├── segment_metadata_N.json
    │   │   │           ├── metadata.json
    │   │   │           └── retrieval
    │   │   │               ├── metadata.json
    │   │   │               ├── segment_metadata_M.json
    │   ├── raw
    │   │   ├── file|products.pdf
    │   │   │   ├── metadata.json
    │   │   │   └── versions
    │   │   │       └── 1
    │   │   │           ├── metadata.json
    │   │   │           └── products.pdf
    │   │   ├── url|arxiv.org_pdf_2505.00661
    │   │   │   ├── metadata.json
    │   │   │   └── versions
    │   │   │       └── 1
    │   │   │           ├── metadata.json
    │   │   │           └── url|arxiv.org_pdf_2505.00661.pdf
    │   │   ├── url|ridgerunai.medium.com_introducing-voice-agent-real-time-voice-assistant-for-language-models-b70fa6f2d593
    │   │   │   ├── metadata.json
    │   │   │   └── versions
    │   │   │       └── 1
    │   │   │           ├── metadata.json
    │   │   │           └── url|ridgerunai.medium.com_introducing-voice-agent-real-time-voice-assistant-for-language-models-b70fa6f2d593.html
    │   ├── segmentation
    │   │   ├── file|products.pdf
    │   │   │   └── versions
    │   │   │       └── 1
    │   │   │           ├── curriculum
    │   │   │           │   └── segment_N.json
    │   │   │           ├── retrieval
    │   │   │           │   └── segment_M.json
    │   │   │           └── segmentation.json
    │   │   ├── url|arxiv.org_pdf_2505.00661
    │   │   │   └── versions
    │   │   │       └── 1
    │   │   │           ├── curriculum
    │   │   │           │   ├── segment_N.json
    │   │   │           ├── retrieval
    │   │   │           │   ├── segment_M.json
    │   │   │           └── segmentation.json
    │   │   ├── url|ridgerunai.medium.com_introducing-voice-agent-real-time-voice-assistant-for-language-models-b70fa6f2d593
    │   │   │   └── versions
    │   │   │       └── 1
    │   │   │           ├── curriculum
    │   │   │           │   ├── segment_N.json
    │   │   │           ├── retrieval
    │   │   │           │   ├── segment_M.json
    │   │   │           └── segmentation.json
    │   ├── summarization
    │   └── tf-idf
    └── log
        └── knowledge_base
            └── 2025
                └── 06
                    ├── 26
                    │   └── operations.jsonl
                    └── 27
                        └── operations.jsonl
```

> This tree of the project organization is built with `tree` utility (`tree .`).
