# agent_core.py

import time
import re
import asyncio
from typing import List, Tuple

class ContentAgent:
    def __init__(self, llm):
        self.llm = llm
        self.reasoning_log = []

    async def _generate_response_async(self, prompt: str) -> str:
        """Async wrapper to call the Ollama LLM."""
        formatted_prompt = f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n"
        response = await self.llm.ainvoke(formatted_prompt)
        return response.strip()

    async def generate_outline(self, user_request: str, title: str) -> List[str]:
        """Generates and parses the document outline asynchronously."""
        prompt = f"""You are an expert content creator. Generate a bullet-point outline for a document:
Title: "{title}"
User request: "{user_request}"

Rules:
- Use hyphens (“-”) at the starts of lines.
- First item must be “Executive Summary”.

Begin:"""

        raw = await self._generate_response_async(prompt)
        outline = [p.strip() for p in re.findall(r'^\s*[-*]\s*(.+)', raw, re.MULTILINE)]
        self.reasoning_log.append({"stage": "outline", "raw": raw, "parsed": outline})
        return outline

    async def generate_section_content(self, topic: str, outline: List[str], title: str) -> Tuple[str, str]:
        """Generates the content for a single section using adaptive prompts."""
        
        # Your excellent adaptive prompt logic
        topic_lower = topic.lower()
        if "conclusion" in topic_lower or "summary" in topic_lower:
            prompt = f"""Write a strong conclusion or summary for the document titled "{title}".
Summarize the key insights from the following outline points:
{chr(10).join(f"- {t}" for t in outline[:5])}

Do not repeat the section title. Begin writing immediately:"""
        elif any(kw in topic_lower for kw in ["prediction", "impact", "ethical", "future", "societal"]):
            prompt = f"""Write the section on “{topic}” for the document "{title}".
Focus on the core ideas in concise, clear paragraphs. Use this minimal context:
{chr(10).join(f"- {t}" for t in outline[:3])}

Begin writing immediately:"""
        else:
            prompt = f"""Write the section for topic: "{topic}" in document "{title}".
Do not repeat the topic as a heading. Begin with content directly:"""

        print(f"[DEBUG] Topic: {topic}")
        print(f"[DEBUG] Prompt length: {len(prompt)} chars")

        content = await self._generate_response_async(prompt)
        return topic, content

    async def _retry_simple(self, topic: str, title: str) -> Tuple[str, str]:
        """A robust fallback method for a failed section generation."""
        print(f"[INFO] Retrying section '{topic}' with a simpler prompt...")
        fallback_prompt = f"""Write a short, clear paragraph about the topic "{topic}" for the document titled "{title}"."""
        try:
            # Use a short timeout for the retry to prevent it from hanging
            content = await asyncio.wait_for(self._generate_response_async(fallback_prompt), timeout=45)
            return topic, content.strip()
        except Exception as e:
            print(f"[ERROR] Retry also failed for '{topic}': {e}")
            return topic, f"**[ERROR: Could not generate content for section '{topic}'. Please try regenerating manually.]**"

    async def safe_generate_with_retry(self, topic: str, outline: List[str], title: str) -> Tuple[str, str]:
        """A safe wrapper that handles timeouts and errors for a single section generation."""
        # Your excellent dynamic timeout logic
        timeout = 120 if any(kw in topic.lower() for kw in ["conclusion", "future", "impact", "ethical"]) else 60

        try:
            # --- THIS IS THE FIX: Pass the 'outline' argument correctly ---
            return await asyncio.wait_for(self.generate_section_content(topic, outline, title), timeout=timeout)
        except asyncio.TimeoutError:
            print(f"[TIMEOUT] Section '{topic}' timed out.")
            return await self._retry_simple(topic, title)
        except Exception as e:
            print(f"[ERROR] Section '{topic}' failed: {e}")
            return await self._retry_simple(topic, title)

    async def run_initial_draft_async(self, user_request: str, document_title: str):
        """Main async pipeline that orchestrates generation and captures full performance metrics."""
        metrics = {"total_time": 0, "outline_time": 0, "total_tokens": 0, "ttft": 0, "inter_token_latency": 0, "tokens_per_second": 0}
        start_all = time.time()
        self.reasoning_log = []

        print("[INFO] Starting outline generation...")
        outline_start = time.time()
        outline = await self.generate_outline(user_request, document_title)
        outline_time = time.time() - outline_start
        metrics["outline_time"] = outline_time
        print(f"[INFO] Outline generated in {outline_time:.2f} seconds: {outline}")

        if not outline:
            raise ValueError("Outline generation failed.")

        # Your excellent concurrency limiting
        semaphore = asyncio.Semaphore(3) # A limit of 3 is a good balance

        async def sem_task(topic):
            async with semaphore:
                print(f"[INFO] Starting section generation for topic: '{topic}'")
                section_start = time.time()
                result = await self.safe_generate_with_retry(topic, outline, document_title)
                section_time = time.time() - section_start
                print(f"[INFO] Finished section '{topic}' in {section_time:.2f} seconds")
                return result

        tasks = [sem_task(topic) for topic in outline]
        print(f"[INFO] Launching {len(tasks)} section generation tasks with concurrency 3...")
        
        start_of_generation = time.time()
        results_list = await asyncio.gather(*tasks)
        end_of_generation = time.time()

        # --- THIS IS THE FIX: The retry logic is now correctly handled within safe_generate ---
        # The flawed, redundant retry block has been removed.
        
        results_dict = dict(results_list)
        sections = [
            {"topic": topic, "content": results_dict.get(topic, "").strip(), "feedback": ""}
            for topic in outline
        ]

        full_text = "".join([s["content"] for s in sections])
        if not full_text.strip():
            raise ValueError("No content was generated.")

        total_tokens = self.llm.get_num_tokens(full_text)
        metrics["total_tokens"] = total_tokens
        total_time = time.time() - start_all
        metrics["total_time"] = total_time
        metrics["ttft"] = metrics["outline_time"]
        #metrics["ttft"] = metrics["outline_time"]
        generation_duration = end_of_generation - start_of_generation
        if total_tokens > 1 and (end_of_generation - start_of_generation) > 0:
            duration = end_of_generation - start_of_generation
            metrics["inter_token_latency"] = (duration - metrics["ttft"]) / (total_tokens - 1) * 1000
            metrics["tokens_per_second"] = total_tokens / generation_duration # t/s

        print(f"[INFO] Document generation complete in {total_time:.2f} seconds, total tokens: {total_tokens}")

        return "\n".join(outline), sections, metrics

    def regenerate_section(self, document_title: str, outline: list, section_topic: str, original_content: str, user_feedback: str) -> str:
        """Regenerates a section based on user feedback."""
        
        # --- THIS IS THE FIX: The variable name 'title' is corrected to 'document_title' ---
        prompt = f"""You are an expert academic editor revising a section based on user feedback.

            DOCUMENT TITLE: "{document_title}"
            FULL OUTLINE (for context):
            {chr(10).join([f'- {t}' for t in outline])}
            SECTION TOPIC: "{section_topic}"

            ORIGINAL DRAFT:
            ---
            {original_content}
            ---

            USER FEEDBACK: "{user_feedback}"

            Provide the new, revised version of the section content. Begin writing the revised content directly:
            REVISED CONTENT:"""

        formatted_prompt = f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n"
        return self.llm.invoke(formatted_prompt)