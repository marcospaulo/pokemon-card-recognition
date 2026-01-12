"""Data structures for Pokemon card recognition pipeline

Defines common data classes used across the pipeline:
- Detection: YOLO detection result with oriented bounding box
- CardMatch: Database match result
- RecognitionResult: Full pipeline output
"""

from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Dict, Any


@dataclass
class Detection:
    """Detection result from YOLO11-OBB

    Represents a detected object (Pokemon card) with oriented bounding box.

    Attributes:
        card_name: Detected class name (e.g., "Pokemon Card")
        confidence: Detection confidence score (0.0-1.0)
        obb: Oriented bounding box (cx, cy, width, height, angle_radians)
    """
    card_name: str
    confidence: float
    obb: Tuple[float, float, float, float, float]  # cx, cy, w, h, angle

    @property
    def center(self) -> Tuple[float, float]:
        """Get bounding box center coordinates"""
        return (self.obb[0], self.obb[1])

    @property
    def size(self) -> Tuple[float, float]:
        """Get bounding box dimensions (width, height)"""
        return (self.obb[2], self.obb[3])

    @property
    def angle(self) -> float:
        """Get rotation angle in radians"""
        return self.obb[4]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'card_name': self.card_name,
            'confidence': self.confidence,
            'obb': {
                'cx': self.obb[0],
                'cy': self.obb[1],
                'width': self.obb[2],
                'height': self.obb[3],
                'angle': self.obb[4]
            }
        }


@dataclass
class CardMatch:
    """Database match result for a card embedding

    Represents a match found in the reference database.

    Attributes:
        card_id: Unique card identifier (e.g., "xy1-1")
        name: Pokemon/card name (e.g., "Venusaur EX")
        set_name: Set name (e.g., "XY Base")
        set_code: Set code (e.g., "xy1")
        similarity: Cosine similarity score (0.0-1.0)
        distance: L2 distance in embedding space
        rank: Match rank (1 = best match)
    """
    card_id: str
    name: str
    set_name: str = ""
    set_code: str = ""
    similarity: float = 0.0
    distance: float = 0.0
    rank: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'card_id': self.card_id,
            'name': self.name,
            'set_name': self.set_name,
            'set_code': self.set_code,
            'similarity': self.similarity,
            'distance': self.distance,
            'rank': self.rank
        }


@dataclass
class RecognitionResult:
    """Full recognition pipeline result

    Contains detection, embedding, and matching results with timing.

    Attributes:
        detection: Optional detection result (None if no card found)
        matches: List of card matches sorted by similarity
        embedding: Extracted embedding vector (768D)
        timing: Performance timing breakdown
        frame_id: Frame number in sequence
        is_valid: Whether a valid card was recognized
    """
    detection: Optional[Detection] = None
    matches: List[CardMatch] = field(default_factory=list)
    embedding: Optional[List[float]] = None
    timing: Dict[str, float] = field(default_factory=dict)
    frame_id: int = 0
    is_valid: bool = False

    @property
    def top_match(self) -> Optional[CardMatch]:
        """Get best match if available"""
        return self.matches[0] if self.matches else None

    @property
    def confidence(self) -> float:
        """Get confidence of top match"""
        return self.matches[0].similarity if self.matches else 0.0

    @property
    def total_time_ms(self) -> float:
        """Get total processing time in milliseconds"""
        return self.timing.get('total_ms', 0.0)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'detection': self.detection.to_dict() if self.detection else None,
            'matches': [m.to_dict() for m in self.matches],
            'timing': self.timing,
            'frame_id': self.frame_id,
            'is_valid': self.is_valid,
            'top_match': self.top_match.to_dict() if self.top_match else None
        }
