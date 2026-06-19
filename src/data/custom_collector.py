"""
Custom Dataset Collector — AI-Powered Lie Detection System
Protocol for recording and labeling custom truth/lie audio+text pairs.
Includes consent form, recording script, and dataset schema.
"""
from __future__ import annotations

import csv
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from src.utils.logger import logger


# ──────────────────────────────────────────────────────────────
# Recording Script (20 Truth + 20 Lie prompts per participant)
# ──────────────────────────────────────────────────────────────

RECORDING_SCRIPT = {
    "truth_prompts": [
        "Tell me your full name and where you were born.",
        "Describe what you had for breakfast this morning.",
        "What is your current job or main daily activity?",
        "Tell me about a hobby you genuinely enjoy.",
        "Describe the last movie or TV show you watched.",
        "What did you do last weekend?",
        "Tell me about your hometown.",
        "Describe a close friend or family member.",
        "What is your favourite season and why?",
        "Tell me about your morning routine.",
        "Describe a trip you have taken in the last two years.",
        "What kind of music do you usually listen to?",
        "Tell me about a skill you have developed recently.",
        "What is your favourite food?",
        "Describe a typical evening at home.",
        "Tell me about a book you have read and enjoyed.",
        "What sport or physical activity do you like most?",
        "Describe your commute to work or school.",
        "What do you do to relax after a stressful day?",
        "Tell me about something you are proud of achieving.",
    ],
    "lie_prompts": [
        "Claim that you are a medical doctor with 10 years experience.",
        "Insist that you visited Paris last month, when you have never been.",
        "Say that you won a national sports competition last year.",
        "Tell me that you own a luxury sports car.",
        "Claim you speak five languages fluently.",
        "Insist that you were on television recently.",
        "Say that you met a famous celebrity last week.",
        "Claim that you cook gourmet meals every evening.",
        "Insist that you read at least two books every week.",
        "Say that you meditate for two hours every morning.",
        "Claim that your salary is over $300,000 per year.",
        "Say that you have climbed Mount Everest.",
        "Insist that you personally know the president or prime minister.",
        "Claim you have never been nervous or anxious in your life.",
        "Say that you won an award for public speaking recently.",
        "Claim that you run a marathon every month.",
        "Insist that you have a photographic memory.",
        "Say that you invented a product that is sold globally.",
        "Claim that you have never made a mistake at work.",
        "Insist that you sleep only three hours a night and feel completely fine.",
    ],
    "instructions": (
        "Read each prompt carefully. Speak naturally and confidently. "
        "Do not pause before speaking. Maintain a neutral posture and face the microphone. "
        "Record each response for 10–30 seconds."
    ),
}


# ──────────────────────────────────────────────────────────────
# Consent Form Template
# ──────────────────────────────────────────────────────────────

CONSENT_FORM_TEMPLATE = """
================================================================================
              INFORMED CONSENT FORM — AI LIE DETECTION RESEARCH STUDY
================================================================================

Study Title  : AI-Powered Deception Detection Using Multimodal Analysis
Research Team: AI Research Laboratory
IRB Protocol : [IRB-XXXX-YYYY]

PURPOSE
-------
You are invited to participate in a research study to develop and evaluate
AI models capable of detecting deception patterns in speech and text.

WHAT YOU WILL DO
----------------
- Read and record 20 truthful statements about yourself (~5 minutes)
- Read and record 20 fabricated statements provided by the researchers (~5 minutes)
- Fill in a demographic questionnaire (optional)

CONFIDENTIALITY
---------------
- Your recordings will be anonymised with a unique Participant ID.
- No personally identifying information will be stored alongside recordings.
- Data will be stored on encrypted servers and accessible only to the research team.
- Data may be published as an anonymised academic dataset.

VOLUNTARY PARTICIPATION
-----------------------
- Participation is completely voluntary.
- You may withdraw at any time without penalty.
- You may request deletion of your data within 30 days of collection.

RISKS & BENEFITS
----------------
- No known physical risks. Mild discomfort from speaking fabricated statements.
- Contribution to academic research on deception detection.

CONTACT
-------
Principal Investigator: [Name] | [email@institution.edu] | [phone]

================================================================================
PARTICIPANT AGREEMENT

I have read and understood the information above. I voluntarily agree to
participate in this study.

Participant Name  : ________________________________
Participant ID    : ________________________________ (assigned by researcher)
Date              : ________________________________
Signature         : ________________________________

Researcher Name   : ________________________________
Date              : ________________________________
Signature         : ________________________________
================================================================================
"""


# ──────────────────────────────────────────────────────────────
# Dataset Schema
# ──────────────────────────────────────────────────────────────

DATASET_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "LieDetectionDatasetRecord",
    "type": "object",
    "required": ["record_id", "participant_id", "label", "text", "audio_path", "timestamp"],
    "properties": {
        "record_id": {"type": "string", "description": "Unique UUID for this record"},
        "participant_id": {"type": "string", "description": "Anonymised participant ID"},
        "session_id": {"type": "string", "description": "Session UUID"},
        "label": {"type": "integer", "enum": [0, 1], "description": "0=truth, 1=lie"},
        "label_str": {"type": "string", "enum": ["truth", "lie"]},
        "text": {"type": "string", "description": "Transcribed or read statement"},
        "prompt": {"type": "string", "description": "Prompt given to participant"},
        "audio_path": {"type": "string", "description": "Relative path to WAV file"},
        "audio_duration_sec": {"type": "number", "description": "Duration in seconds"},
        "sample_rate": {"type": "integer", "default": 22050},
        "timestamp": {"type": "string", "format": "date-time"},
        "metadata": {
            "type": "object",
            "properties": {
                "age_group": {"type": "string", "enum": ["18-25", "26-35", "36-50", "50+"]},
                "gender": {"type": "string", "enum": ["male", "female", "non-binary", "prefer_not_to_say"]},
                "native_language": {"type": "string"},
                "recording_device": {"type": "string"},
                "room_noise_level": {"type": "string", "enum": ["quiet", "moderate", "noisy"]},
                "collection_method": {"type": "string", "enum": ["lab", "remote", "crowdsource"]},
            }
        },
        "quality_flags": {
            "type": "object",
            "properties": {
                "clipping": {"type": "boolean"},
                "background_noise": {"type": "boolean"},
                "too_short": {"type": "boolean"},
                "too_long": {"type": "boolean"},
                "accepted": {"type": "boolean"},
            }
        }
    }
}


# ──────────────────────────────────────────────────────────────
# Data Collection Protocol Manager
# ──────────────────────────────────────────────────────────────

class CustomDataCollector:
    """
    Manages the custom dataset collection protocol.
    Handles participant registration, session management, and record writing.
    """

    def __init__(self, output_dir: str = "data/custom"):
        self.output_dir = Path(output_dir)
        self.audio_dir = self.output_dir / "audio"
        self.records_file = self.output_dir / "records.jsonl"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.audio_dir.mkdir(parents=True, exist_ok=True)

    def register_participant(
        self,
        age_group: str = "18-25",
        gender: str = "prefer_not_to_say",
        native_language: str = "English",
    ) -> str:
        """Register a new participant and return their ID."""
        participant_id = f"P{uuid.uuid4().hex[:8].upper()}"
        logger.info(f"Registered participant: {participant_id}")
        return participant_id

    def create_record(
        self,
        participant_id: str,
        label: int,
        text: str,
        prompt: str,
        audio_path: str,
        audio_duration_sec: float,
        metadata: Optional[Dict] = None,
        quality_flags: Optional[Dict] = None,
    ) -> Dict:
        """Create a dataset record conforming to DATASET_SCHEMA."""
        record = {
            "record_id": str(uuid.uuid4()),
            "participant_id": participant_id,
            "session_id": str(uuid.uuid4()),
            "label": label,
            "label_str": "lie" if label == 1 else "truth",
            "text": text,
            "prompt": prompt,
            "audio_path": audio_path,
            "audio_duration_sec": audio_duration_sec,
            "sample_rate": 22050,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "metadata": metadata or {},
            "quality_flags": quality_flags or {
                "clipping": False,
                "background_noise": False,
                "too_short": audio_duration_sec < 1.0,
                "too_long": audio_duration_sec > 60.0,
                "accepted": True,
            },
        }
        return record

    def save_record(self, record: Dict) -> None:
        """Append a record to the JSONL file."""
        with open(self.records_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def load_records(self) -> List[Dict]:
        """Load all collected records."""
        if not self.records_file.exists():
            return []
        records = []
        with open(self.records_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def get_statistics(self) -> Dict:
        """Return collection statistics."""
        records = self.load_records()
        if not records:
            return {"total": 0}
        import pandas as pd
        df = pd.DataFrame(records)
        return {
            "total": len(df),
            "truth_count": (df["label"] == 0).sum(),
            "lie_count": (df["label"] == 1).sum(),
            "participants": df["participant_id"].nunique(),
            "avg_duration_sec": df["audio_duration_sec"].mean(),
            "accepted": df.apply(
                lambda r: r.get("quality_flags", {}).get("accepted", True), axis=1
            ).sum(),
        }

    def export_schema(self, output_path: str = "data/schemas/dataset_schema.json") -> None:
        """Write the dataset JSON schema to disk."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(DATASET_SCHEMA, f, indent=2)
        logger.info(f"Dataset schema written to {path}")

    def export_consent_form(self, output_path: str = "docs/consent_form.txt") -> None:
        """Write the consent form template to disk."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(CONSENT_FORM_TEMPLATE)
        logger.info(f"Consent form written to {path}")

    def export_recording_script(
        self, output_path: str = "docs/recording_script.json"
    ) -> None:
        """Write the recording script to disk."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(RECORDING_SCRIPT, f, indent=2)
        logger.info(f"Recording script written to {path}")
