"""Focused tests for upload indexing and chat-scoped retrieval."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import faiss
import numpy as np

from backend.rag.pipeline import AuraRAG, _find_demo
from backend.rag import generator
from backend.rag.document_loader import load_pdf
from backend.rag.guardrails import validate_retrieval
from backend.rag.user_store import UserDocumentStore
from scripts.benchmark_e2e import load_queries


class FixedEmbeddings:
    def encode(self, texts, **_kwargs):
        return np.tile(np.array([[1.0, 0.0]], dtype="float32"), (len(texts), 1))


class RagWorkflowTests(unittest.TestCase):
    @staticmethod
    def write_minimal_pdf(path: Path, text: str) -> None:
        stream = f"BT /F1 12 Tf 20 80 Td ({text}) Tj ET".encode("ascii")
        objects = [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            (
                b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 144] "
                b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>"
            ),
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
            b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n"
            + stream
            + b"\nendstream",
        ]
        data = bytearray(b"%PDF-1.4\n")
        offsets = [0]
        for object_number, body in enumerate(objects, start=1):
            offsets.append(len(data))
            data.extend(f"{object_number} 0 obj\n".encode("ascii"))
            data.extend(body + b"\nendobj\n")

        xref_offset = len(data)
        data.extend(f"xref\n0 {len(offsets)}\n".encode("ascii"))
        data.extend(b"0000000000 65535 f \n")
        for offset in offsets[1:]:
            data.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
        data.extend(
            (
                f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\n"
                f"startxref\n{xref_offset}\n%%EOF\n"
            ).encode("ascii")
        )
        path.write_bytes(data)

    def make_rag(self, search):
        rag = AuraRAG.__new__(AuraRAG)
        rag.index = faiss.IndexFlatIP(2)
        rag.metadata = []
        rag.chunk_count = 0
        rag.user_store = Mock()
        rag.user_store.size = 1
        rag.user_store.search = search
        return rag

    def test_empty_chat_attachment_scope_never_searches_global_uploads(self):
        search = Mock(return_value=[])
        rag = self.make_rag(search)

        rag._retrieve_once(
            "What does the file say?",
            use_base_corpus=False,
            use_uploads=True,
            session_attachments=[],
        )

        search.assert_not_called()

    def test_chat_retrieval_is_limited_to_its_attached_documents(self):
        search = Mock(return_value=[])
        rag = self.make_rag(search)

        rag._retrieve_once(
            "Summarize the document",
            use_base_corpus=False,
            use_uploads=True,
            session_attachments=[{"doc_id": "chat-document"}],
        )

        search.assert_called_once()
        self.assertEqual(search.call_args.kwargs["doc_ids"], ["chat-document"])

    def test_uploaded_text_is_indexed_and_survives_store_reload(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            upload = root / "notes.txt"
            upload.write_text(
                "AURA stores uploaded documents as searchable evidence. "
                "This sentence is long enough to create an indexable chunk.",
                encoding="utf-8",
            )

            paths = {
                "USER_INDEX_DIR": root / "index",
                "USER_INDEX_FILE": root / "index" / "user.index",
                "USER_METADATA_FILE": root / "index" / "user_metadata.jsonl",
            }
            with patch.multiple("backend.rag.user_store", **paths):
                store = UserDocumentStore(FixedEmbeddings(), dim=2)
                info = store.ingest_file(upload, original_name="notes.txt")
                hits = store.search(
                    "searchable evidence",
                    doc_ids=[info["doc_id"]],
                )
                other_chat_hits = store.search(
                    "searchable evidence",
                    doc_ids=["another-chat-document"],
                )
                reloaded = UserDocumentStore(FixedEmbeddings(), dim=2)

            self.assertGreater(info["chunks"], 0)
            self.assertEqual(hits[0]["source_file"], "notes.txt")
            self.assertEqual(other_chat_hits, [])
            self.assertEqual(reloaded.size, info["index_size"])
            self.assertEqual(reloaded.list_documents()[0]["doc_id"], info["doc_id"])
            removed_chunks = reloaded.remove_documents([info["doc_id"]])
            self.assertEqual(removed_chunks, info["chunks"])
            self.assertEqual(reloaded.size, 0)
            self.assertEqual(
                UserDocumentStore(FixedEmbeddings(), dim=2).size,
                0,
            )

    def test_pdf_text_is_extracted_for_upload_ingestion(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "evidence.pdf"
            self.write_minimal_pdf(path, "PDF upload pipeline smoke evidence.")
            self.assertIn("PDF upload pipeline smoke evidence.", load_pdf(path))

    def test_local_generation_uses_loopback_ollama_and_preserves_code_formatting(self):
        response = Mock()
        response.json.return_value = {
            "message": {"content": "Project draft:\n\n```python\nprint('hello')\n```"}
        }
        client = MagicMock()
        client.__enter__.return_value = client
        client.post.return_value = response
        evidence = [
            {
                "text": "A local project must be grounded in this documentation.",
                "source_file": "guide.pdf",
                "chunk_index": 2,
                "final_score": 0.9,
            }
        ]

        with (
            patch.object(generator, "LOCAL_LLM_MODEL", "qwen2.5:3b"),
            patch.object(generator, "LOCAL_LLM_URL", "http://127.0.0.1:11434"),
            patch.object(generator.httpx, "Client", return_value=client),
        ):
            result = generator.generate_answer("Draft a project", evidence)

        self.assertIn("```python\nprint('hello')\n```", result["answer"])
        self.assertEqual(result["generation_mode"], "ollama")
        self.assertIn("[guide.pdf, chunk 2]", client.post.call_args.kwargs["json"]["messages"][1]["content"])

    def test_local_generation_rejects_non_loopback_urls(self):
        with (
            patch.object(generator, "LOCAL_LLM_MODEL", "qwen2.5:3b"),
            patch.object(generator, "LOCAL_LLM_URL", "https://example.com"),
            patch.object(generator.httpx, "Client") as client,
        ):
            with self.assertRaisesRegex(RuntimeError, "loopback"):
                generator.generate_answer(
                    "Explain this",
                    [{"text": "Relevant evidence", "final_score": 0.9}],
                )
        client.assert_not_called()

    def test_unsafe_queries_skip_local_generation(self):
        rag = self.make_rag(Mock(return_value=[]))
        rag.chunk_count = 1
        rag.user_store.size = 0
        rag.retrieve = Mock(
            return_value=(
                [
                    {
                        "text": "retrieved text",
                        "final_score": 0.9,
                        "semantic_score": 0.9,
                    }
                ],
                {},
                {},
            )
        )
        with patch("backend.rag.pipeline.generate_answer") as generate:
            result = rag.ask("How to make a bomb?")
        generate.assert_not_called()
        self.assertEqual(result["reason"], "unsafe_query")
        self.assertEqual(result["generation_mode"], "guardrail")

    def test_upload_evidence_has_a_relevance_gate(self):
        self.assertTrue(
            validate_retrieval(
                [
                    {
                        "origin": "upload",
                        "final_score": 0.46,
                        "semantic_score": 0.70,
                        "lexical_score": 0.5,
                    }
                ]
            )
        )
        self.assertFalse(
            validate_retrieval(
                [
                    {
                        "origin": "upload",
                        "final_score": 0.43,
                        "semantic_score": 0.61,
                        "lexical_score": 0.1,
                    }
                ]
            )
        )

    def test_uploaded_extractive_answer_selects_relevant_passage(self):
        result = generator.generate_answer(
            "What is the default maximum upload size stated in the attached README?",
            [
                {
                    "origin": "upload",
                    "final_score": 0.49,
                    "semantic_score": 0.77,
                    "lexical_score": 0.5,
                    "text": (
                        "Chat attachments are isolated between sessions. "
                        "Upload size is limited by AURA_MAX_UPLOAD_MB "
                        "(25 MB by default). Pipeline stages are documented below."
                    ),
                    "answer": (
                        "Chat attachments are isolated between sessions. "
                        "Upload size is limited by AURA_MAX_UPLOAD_MB "
                        "(25 MB by default). Pipeline stages are documented below."
                    ),
                }
            ],
        )
        self.assertTrue(result["grounded"])
        self.assertEqual(
            result["answer"],
            "Upload size is limited by AURA_MAX_UPLOAD_MB (25 MB by default).",
        )

    def test_benchmark_fallback_queries_exercise_retrieval(self):
        with patch("scripts.benchmark_e2e.os.path.exists", return_value=False):
            queries = load_queries(6)
        self.assertEqual(len(queries), 6)
        self.assertTrue(all(_find_demo(query) is None for query in queries))


if __name__ == "__main__":
    unittest.main()
