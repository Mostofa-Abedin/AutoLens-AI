from pydantic import BaseModel, ConfigDict
from typing import Optional, List
from enum import Enum


class EvaluationMode(str, Enum):
    cold = "cold"
    rag = "rag"


class Provider(str, Enum):
    anthropic = "anthropic"
    google = "google"
    openai = "openai"
    ollama = "ollama"


class FileNode(BaseModel):
    name: str
    type: str  # "folder" or "image"
    path: Optional[str] = None
    thumb_url: Optional[str] = None
    children: Optional[List["FileNode"]] = None
    image_count: Optional[int] = None


FileNode.model_rebuild()


class BrowseTreeResponse(BaseModel):
    tree: List[FileNode]
    total_images: int
    root_path: str


class SimilarImage(BaseModel):
    path: str
    thumb_url: str
    class_id: str
    angle: str
    similarity_score: float


class IdentifyRequest(BaseModel):
    image_path: str
    provider: Provider
    model: str
    api_key: str
    ollama_endpoint: Optional[str] = "http://localhost:11434"


class IdentifyResponse(BaseModel):
    make: str
    model: str
    year_estimate: str
    confidence: str
    confidence_score: float
    reasoning: str
    in_database: bool
    knowledge_base_id: Optional[str] = None
    knowledge_base_label: Optional[str] = None


class EvaluateRequest(BaseModel):
    image_path: str
    mode: EvaluationMode
    provider: Provider
    model: str
    api_key: str
    prompt: str
    ollama_endpoint: Optional[str] = "http://localhost:11434"
    knowledge_base_id: Optional[str] = None


class EvaluateResponse(BaseModel):
    class_id: str
    class_label: str
    confidence: str  # "high" | "medium" | "low"
    confidence_score: float
    reasoning: str
    visible_features: List[str]
    similar_images: Optional[List[SimilarImage]] = None


class PromptCreate(BaseModel):
    name: str
    content: str
    is_default: bool = False


class PromptUpdate(BaseModel):
    name: Optional[str] = None
    content: Optional[str] = None
    is_default: Optional[bool] = None


class PromptResponse(BaseModel):
    id: int
    name: str
    content: str
    is_default: bool
    created_at: str


class SessionResultCreate(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    image_path: str
    image_filename: str
    predicted_class: str
    actual_class: str
    correct: bool
    mode: EvaluationMode
    model_used: str
    confidence_score: float


class SessionResultRequest(SessionResultCreate):
    session_id: str


class ClassStat(BaseModel):
    class_id: str
    class_label: str
    total: int
    correct: int
    accuracy: float


class SessionStats(BaseModel):
    session_id: str
    total: int
    correct: int
    accuracy: float
    by_class: List[ClassStat]


class IndexStatusResponse(BaseModel):
    built: bool
    total_images: int
    index_path: str
    last_built: Optional[str] = None


class IndexEntry(BaseModel):
    filename: str
    path: str
    angle: str


class IndexClassGroup(BaseModel):
    class_id: str
    count: int
    entries: List[IndexEntry]


class IndexInspectResponse(BaseModel):
    total: int
    by_class: List[IndexClassGroup]


class BuildIndexRequest(BaseModel):
    dataset_path: str
