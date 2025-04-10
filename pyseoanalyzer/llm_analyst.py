from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_community.chat_models import ChatLiteLLM # For LiteLLM Proxy
from langchain_ollama import ChatOllama # For direct Ollama connection
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, AliasChoices # Import AliasChoices
from typing import Dict, List, Optional

import asyncio
import json
import os
import logging

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Pydantic models for structured output
class EntityAnalysis(BaseModel):
    entity_assessment: str = Field(
        description="Detailed analysis of entity optimization"
    )
    knowledge_panel_readiness: int = Field(description="Score from 0-100")
    key_improvements: List[str] = Field(description="Top 3 improvements needed")


from typing import Dict, List, Optional, Union # Add Union

class CredibilityAnalysis(BaseModel):
    credibility_assessment: str = Field(description="Overall credibility analysis")
    # Allow aliases and potentially nested dict for flexibility with LLM output
    neeat_scores: Dict[str, Union[int, Dict]] = Field(
        description="Individual N-E-E-A-T-U component scores (0-100 integer)", # Added type hint
        validation_alias=AliasChoices('neeat_scores', 'N-E-E-A-T-U', 'neeat_signals')
    )
    trust_signals: List[str] = Field(description="Identified trust signals")


class ConversationAnalysis(BaseModel):
    conversation_readiness: str = Field(description="Overall assessment")
    query_patterns: List[str] = Field(description="Identified query patterns")
    engagement_score: int = Field(description="Score from 0-100")
    gaps: List[str] = Field(description="Identified conversational gaps", validation_alias=AliasChoices('gaps', 'gap_analysis', 'gap_s')) # Added alias


class PlatformPresence(BaseModel):
    platform_coverage: Dict[str, str] = Field(
        description="Coverage analysis per platform"
    )
    visibility_scores: Dict[str, int] = Field(description="Scores per platform type")
    optimization_opportunities: List[str] = Field(description="List of opportunities")


class SEORecommendations(BaseModel):
    strategic_recommendations: List[str] = Field(
        description="Major strategic recommendations"
    )
    quick_wins: List[str] = Field(description="Immediate action items")
    long_term_strategy: List[str] = Field(description="Long-term strategic goals")
    priority_matrix: Dict[str, List[str]] = Field(
        description="Priority matrix by impact/effort"
    )


class LLMSEOEnhancer:
    def __init__(self):
        # Explicitly reload .env within __init__ to ensure latest values are read
        dotenv_path = os.path.join(os.path.dirname(__file__), '..','..','..', '.env') # Path relative to llm_analyst.py
        load_dotenv(dotenv_path=dotenv_path, override=True)
        logging.debug(f"Reloaded .env file from: {dotenv_path}")

        self.llm_provider = os.environ.get("LLM_PROVIDER", "anthropic").lower() # Store provider name
        llm_model = os.environ.get("LLM_MODEL") # Specific model name
        logging.debug(f"Read LLM_MODEL from env: {llm_model}") # Add debug log
        temperature = float(os.environ.get("LLM_TEMPERATURE", 0.0))
        timeout = int(os.environ.get("LLM_TIMEOUT", 60)) # Increased timeout
        max_retries = int(os.environ.get("LLM_MAX_RETRIES", 3))

        logging.info(f"Initializing LLM Enhancer with provider: {self.llm_provider}")

        if self.llm_provider == "anthropic":
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                logging.error("ANTHROPIC_API_KEY not found in environment variables.")
                raise ValueError("ANTHROPIC_API_KEY is required for Anthropic provider")
            if not llm_model:
                llm_model = "claude-3-sonnet-20240229" # Default Anthropic model
                logging.warning(f"LLM_MODEL not set, defaulting to {llm_model}")
            self.llm = ChatAnthropic(
                model=llm_model,
                anthropic_api_key=api_key,
                temperature=temperature,
                timeout=timeout,
                max_retries=max_retries,
            )
            logging.info(f"Using Anthropic model: {llm_model}")
        elif self.llm_provider == "litellm":
            # Note: LiteLLM might not support with_structured_output in the same way
            api_base = os.environ.get("LITELLM_API_BASE")
            api_key = os.environ.get("LITELLM_API_KEY", "nokey") # Often not needed for local proxy
            if not api_base:
                logging.error("LITELLM_API_BASE not found in environment variables.")
                raise ValueError("LITELLM_API_BASE is required for LiteLLM provider")
            if not llm_model:
                logging.error("LLM_MODEL must be specified for LiteLLM provider (e.g., 'ollama/mistral').")
                raise ValueError("LLM_MODEL is required for LiteLLM provider")
            self.llm = ChatLiteLLM(
                model=llm_model,
                api_base=api_base,
                api_key=api_key,
                temperature=temperature,
                request_timeout=timeout, # Note: parameter name might differ
                max_retries=max_retries,
            )
            logging.info(f"Using LiteLLM model: {llm_model} via base: {api_base}")
        elif self.llm_provider == "ollama":
            # Provide a default value for OLLAMA_BASE_URL if not set in env
            base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
            if not llm_model:
                logging.error("LLM_MODEL must be specified for Ollama provider (e.g., 'mistral').")
                raise ValueError("LLM_MODEL is required for Ollama provider")
            self.llm = ChatOllama(
                base_url=base_url,
                model=llm_model,
                temperature=temperature,
                request_timeout=120, # Increased timeout to 120 seconds
                # format="json", # Let with_structured_output handle format
            )
            logging.info(f"Using Ollama model: {llm_model} via base: {base_url}") # Removed format=json log
        else:
            # Raise error immediately for unsupported provider
            error_msg = f"Unsupported LLM_PROVIDER: '{self.llm_provider}'. Supported providers: anthropic, litellm, ollama"
            logging.error(error_msg)
            raise ValueError(error_msg)

        self._setup_chains()

    def _setup_chains(self):
        """Setup modern LangChain runnable sequences using pipe syntax"""
        # Entity Analysis Chain
        entity_parser = PydanticOutputParser(pydantic_object=EntityAnalysis)
        entity_prompt = PromptTemplate.from_template(
            """Analyze these SEO elements for entity optimization:
            1. Entity understanding (Knowledge Panel readiness)
            2. Brand credibility signals (N-E-E-A-T-U principles)
            3. Entity relationships and mentions
            4. Topic entity connections
            5. Schema markup effectiveness

            Data to analyze:
            {seo_data}

            {format_instructions}

            Example JSON Output:
            ```json
            {{
              "entity_assessment": "The entity 'Example HVAC Co' shows moderate optimization. Clearer branding and structured data would improve Knowledge Panel readiness.",
              "knowledge_panel_readiness": 55,
              "key_improvements": [
                "Implement Organization schema markup.",
                "Ensure consistent NAP across citations.",
                "Develop a comprehensive 'About Us' page."
              ]
            }}
            ```

            IMPORTANT: Your response MUST be ONLY the JSON object itself, with NO additional text, explanations, or markdown formatting like ```json before or after the JSON object. Ensure all required fields (entity_assessment, knowledge_panel_readiness, key_improvements) are present.
            """
        )
        entity_base_chain = (
            {
                "seo_data": RunnablePassthrough(),
                "format_instructions": lambda _: entity_parser.get_format_instructions(),
            }
            | entity_prompt
        )
        if isinstance(self.llm, ChatOllama):
             logging.info("Using Ollama with_structured_output for entity_chain")
             self.entity_chain = entity_base_chain | self.llm.with_structured_output(EntityAnalysis)
        else:
             self.entity_chain = entity_base_chain | self.llm | entity_parser


        # Credibility Analysis Chain
        credibility_parser = PydanticOutputParser(pydantic_object=CredibilityAnalysis)
        credibility_prompt = PromptTemplate.from_template(
            """Evaluate these credibility aspects:
            1. N-E-E-A-T-U signals
            2. Entity understanding and validation
            3. Content creator credentials
            4. Publisher authority
            5. Topic expertise signals

            Data to analyze:
            {seo_data}

            {format_instructions}

            Example JSON Output:
            ```json
            {{
              "credibility_assessment": "Credibility is moderate. Strong expertise is shown, but author credentials and publisher authority could be clearer.",
              "neeat_scores": {{
                "Experience": 70,
                "Expertise": 85,
                "Authoritativeness": 60,
                "Trustworthiness": 75,
                "Usefulness": 80
              }},
              "trust_signals": [
                "Detailed technical explanations",
                "References to industry standards",
                "Clear contact information (if present)"
              ]
            }}
            ```

            IMPORTANT: Your response MUST be ONLY the JSON object itself, with NO additional text, explanations, or markdown formatting like ```json before or after the JSON object. Ensure all required fields (credibility_assessment, neeat_scores, trust_signals) are present. The values within the `neeat_scores` dictionary MUST be integers (e.g., between 0 and 100), not descriptive strings.
            """
        )
        credibility_base_chain = (
             {
                 "seo_data": RunnablePassthrough(),
                 "format_instructions": lambda _: credibility_parser.get_format_instructions(),
             }
             | credibility_prompt
        )
        if isinstance(self.llm, ChatOllama):
             logging.info("Using Ollama with_structured_output for credibility_chain")
             self.credibility_chain = credibility_base_chain | self.llm.with_structured_output(CredibilityAnalysis)
        else:
             self.credibility_chain = credibility_base_chain | self.llm | credibility_parser


        # Conversation Analysis Chain
        conversation_parser = PydanticOutputParser(pydantic_object=ConversationAnalysis)
        conversation_prompt = PromptTemplate.from_template(
            """Analyze content for conversational search readiness:
            1. Query pattern matching
            2. Intent coverage across funnel
            3. Natural language understanding
            4. Follow-up content availability
            5. Conversational triggers

            Data to analyze:
            {seo_data}

            {format_instructions}

            Example JSON Output:
            ```json
            {{
              "conversation_readiness": "The content answers specific questions well but lacks conversational flow and follow-up prompts.",
              "query_patterns": [
                "how to troubleshoot X",
                "what is Y component",
                "best practices for Z"
              ],
              "engagement_score": 65,
              "gaps": [
                "No clear conversational entry points.",
                "Missing links to related deeper-dive topics.",
                "Lack of interactive elements."
              ]
            }}
            ```

            IMPORTANT: Your response MUST be ONLY the JSON object itself, with NO additional text, explanations, or markdown formatting like ```json before or after the JSON object. Ensure all required fields (conversation_readiness, query_patterns, engagement_score, gaps) are present. Pay close attention to the field name 'gaps'.
            """
        )
        conversation_base_chain = (
             {
                 "seo_data": RunnablePassthrough(),
                 "format_instructions": lambda _: conversation_parser.get_format_instructions(),
             }
             | conversation_prompt
        )
        if isinstance(self.llm, ChatOllama):
             logging.info("Using Ollama with_structured_output for conversation_chain")
             self.conversation_chain = conversation_base_chain | self.llm.with_structured_output(ConversationAnalysis)
        else:
             self.conversation_chain = conversation_base_chain | self.llm | conversation_parser


        # Platform Presence Chain
        platform_parser = PydanticOutputParser(pydantic_object=PlatformPresence)
        platform_prompt = PromptTemplate.from_template(
            """Analyze presence across different platforms:
            1. Search engines (Google, Bing)
            2. Knowledge graphs
            3. AI platforms (ChatGPT, Bard)
            4. Social platforms
            5. Industry-specific platforms

            Data to analyze:
            {seo_data}

            {format_instructions}

            Example JSON Output:
            ```json
            {{
              "platform_coverage": {{
                "Google Search": "High visibility for core terms.",
                "Google Knowledge Graph": "Partial coverage, missing key attributes.",
                "ChatGPT": "Content is discoverable but not optimized for conversational answers.",
                "LinkedIn": "Limited presence.",
                "Industry Forums": "Active participation noted."
              }},
              "visibility_scores": {{
                "Search": 80,
                "Knowledge Graph": 40,
                "AI": 50,
                "Social": 20,
                "Industry": 70
              }},
              "optimization_opportunities": [
                "Optimize content for featured snippets on Google.",
                "Build out LinkedIn profile and share relevant content.",
                "Structure answers for better AI platform consumption."
              ]
            }}
            ```

            IMPORTANT: Your response MUST be ONLY the JSON object itself, with NO additional text, explanations, or markdown formatting like ```json before or after the JSON object. Ensure all required fields (platform_coverage, visibility_scores, optimization_opportunities) are present.
            """
        )
        platform_base_chain = (
             {
                 "seo_data": RunnablePassthrough(),
                 "format_instructions": lambda _: platform_parser.get_format_instructions(),
             }
             | platform_prompt
        )
        if isinstance(self.llm, ChatOllama):
             logging.info("Using Ollama with_structured_output for platform_chain")
             self.platform_chain = platform_base_chain | self.llm.with_structured_output(PlatformPresence)
        else:
             self.platform_chain = platform_base_chain | self.llm | platform_parser


        # Recommendations Chain
        recommendations_parser = PydanticOutputParser(
            pydantic_object=SEORecommendations
        )
        recommendations_prompt = PromptTemplate.from_template(
            """Based on this complete analysis, provide strategic recommendations:
            1. Entity optimization strategy
            2. Content strategy across platforms
            3. Credibility building actions
            4. Conversational optimization
            5. Cross-platform presence improvement

            Analysis results:
            {analysis_results}

            {format_instructions}

            Example JSON Output:
            ```json
            {{
              "strategic_recommendations": [
                "Focus on building topical authority around 'furnace maintenance'.",
                "Develop a content hub for 'HVAC troubleshooting guides'.",
                "Improve E-E-A-T signals by adding author bios and credentials."
              ],
              "quick_wins": [
                "Add 'Organization' schema markup to the homepage.",
                "Optimize title tags for target keywords on top 5 pages.",
                "Ensure all images have descriptive alt text."
              ],
              "long_term_strategy": [
                "Become the leading online resource for DIY HVAC repair.",
                "Expand content to cover commercial HVAC systems.",
                "Build partnerships with HVAC parts suppliers."
              ],
              "priority_matrix": {{
                "High Impact / Low Effort": ["Optimize title tags", "Add image alt text"],
                "High Impact / High Effort": ["Build content hub", "Add author bios"],
                "Low Impact / Low Effort": ["Update footer copyright year"],
                "Low Impact / High Effort": ["Redesign entire website"]
              }}
            }}
            ```

            IMPORTANT: Your response MUST be ONLY the JSON object itself, with NO additional text, explanations, or markdown formatting like ```json before or after the JSON object. Ensure all required fields (strategic_recommendations, quick_wins, long_term_strategy, priority_matrix) are present and the `priority_matrix` keys are exactly as shown in the example.
            """
        )
        recommendations_base_chain = (
             {
                 "analysis_results": RunnablePassthrough(),
                 "format_instructions": lambda _: recommendations_parser.get_format_instructions(),
             }
             | recommendations_prompt
        )
        if isinstance(self.llm, ChatOllama):
             logging.info("Using Ollama with_structured_output for recommendations_chain")
             self.recommendations_chain = recommendations_base_chain | self.llm.with_structured_output(SEORecommendations)
        else:
             self.recommendations_chain = recommendations_base_chain | self.llm | recommendations_parser


        # Store the parser for use in analyze_markdown
        self.recommendations_parser = recommendations_parser


    async def analyze_markdown(self, markdown_content: str) -> Dict:
        """
        Performs LLM analysis directly on provided Markdown content to generate SEO recommendations.
        This is a fallback method and does not perform the full parallel analysis.
        """
        logging.info("Running LLM analysis on provided Markdown content (fallback mode).")

        # Simplified prompt for Markdown analysis -> Recommendations
        markdown_prompt_template = PromptTemplate.from_template(
            """Analyze the following Markdown content extracted from a webpage and generate SEO recommendations.
            Focus on identifying potential SEO improvements based *only* on this text content.
            Consider aspects like keyword usage, topic coverage, clarity, structure, and potential calls-to-action.

            Markdown Content:
            ```markdown
            {markdown_content}
            ```

            {format_instructions}

            Example JSON Output:
            ```json
            {{
              "strategic_recommendations": [
                "Expand on the 'benefits of X' section with more detail.",
                "Consider adding a comparison table for product features.",
                "Target the keyword 'advanced Y techniques' more explicitly."
              ],
              "quick_wins": [
                "Add internal links to related blog posts.",
                "Clarify the main heading for better focus.",
                "Include a stronger call-to-action at the end."
              ],
              "long_term_strategy": [
                "Develop a series of articles around the core topic.",
                "Build out a glossary of technical terms used.",
                "Create downloadable guides based on the content."
              ],
              "priority_matrix": {{
                "High Impact / Low Effort": ["Clarify main heading", "Add internal links"],
                "High Impact / High Effort": ["Expand 'benefits of X' section", "Develop article series"],
                "Low Impact / Low Effort": ["Fix typos"],
                "Low Impact / High Effort": ["Translate content to Spanish"]
              }}
            }}
            ```

            IMPORTANT: Your response MUST be ONLY the JSON object itself, with NO additional text, explanations, or markdown formatting like ```json before or after the JSON object. Ensure all required fields (strategic_recommendations, quick_wins, long_term_strategy, priority_matrix) are present and the `priority_matrix` keys are exactly as shown in the example.
            """
        )

        # Use the existing recommendations_parser or with_structured_output
        markdown_base_chain = (
            {
                "markdown_content": RunnablePassthrough(),
                "format_instructions": lambda _: self.recommendations_parser.get_format_instructions(),
            }
            | markdown_prompt_template
        )

        if isinstance(self.llm, ChatOllama):
             logging.info("Using Ollama with_structured_output for analyze_markdown")
             markdown_chain = markdown_base_chain | self.llm.with_structured_output(SEORecommendations)
        else:
             markdown_chain = markdown_base_chain | self.llm | self.recommendations_parser


        try:
            # If using with_structured_output, the result should already be the parsed Pydantic object
            # If using the parser, the result needs parsing. Langchain might handle this implicitly.
            # Let's assume ainvoke returns the parsed object in both cases for simplicity now.
            recommendations = await markdown_chain.ainvoke(markdown_content)

            # Check if the result is already a dict (from model_dump) or needs dumping
            if isinstance(recommendations, dict):
                 return recommendations
            elif hasattr(recommendations, 'model_dump'):
                 return recommendations.model_dump()
            else:
                 # Handle unexpected return type
                 logging.error(f"Unexpected result type from markdown_chain: {type(recommendations)}")
                 return {"error": "LLM analysis returned unexpected data type."}

        except Exception as e:
            logging.error(f"Error during Markdown LLM analysis: {e}", exc_info=True)
            # Return a dictionary indicating failure
            return {
                "strategic_recommendations": [],
                "quick_wins": [],
                "long_term_strategy": [],
                "priority_matrix": {},
                "error": f"LLM analysis failed: {e}"
            }


    async def enhance_seo_analysis(self, parsed_html_data: Dict) -> Dict: # Ensure async
        """
        Enhanced SEO analysis using modern LangChain patterns based on parsed HTML data.
        """
        # Convert parsed_html_data to string for prompt insertion
        seo_data_str = json.dumps(parsed_html_data, indent=2)

        logging.info("Running parallel LLM chains for full SEO analysis.")
        tasks = [
            self.entity_chain.ainvoke(seo_data_str),
            self.credibility_chain.ainvoke(seo_data_str),
            self.conversation_chain.ainvoke(seo_data_str),
            self.platform_chain.ainvoke(seo_data_str),
        ]

        # Initialize results structure
        final_results = {
            "entity_analysis": None,
            "credibility_analysis": None,
            "conversation_analysis": None,
            "cross_platform_presence": None,
            "recommendations": None,
            "errors": [] # Store specific errors encountered
        }
        valid_analysis_for_recommendations = {}
        analysis_keys = [
            "entity_analysis",
            "credibility_analysis",
            "conversation_analysis",
            "cross_platform_presence",
        ]

        try:
            # Use return_exceptions=True to capture errors without stopping gather
            gather_results = await asyncio.gather(*tasks, return_exceptions=True) # Use await

            for key, result in zip(analysis_keys, gather_results):
                if isinstance(result, Exception):
                    error_msg = f"Error in chain '{key}': {result}"
                    logging.error(error_msg)
                    final_results["errors"].append(error_msg)
                    final_results[key] = {"error": str(result)} # Store error info for this key
                elif result:
                    # Result should be the Pydantic model instance if with_structured_output worked,
                    # or already parsed if using the parser chain.
                    if isinstance(result, dict):
                         final_results[key] = result # Already a dict (e.g., from non-Ollama parser)
                    elif hasattr(result, 'model_dump'):
                         final_results[key] = result.model_dump() # Dump Pydantic model
                    else:
                         # Handle unexpected successful result type
                         logging.warning(f"Chain '{key}' returned unexpected type: {type(result)}")
                         final_results[key] = {"error": f"Chain '{key}' returned unexpected data type."}
                         final_results["errors"].append(f"Chain '{key}' returned unexpected data type.")
                         continue # Skip adding to valid_analysis

                    # Add valid results for recommendations input if no error stored
                    if "error" not in final_results[key]:
                         valid_analysis_for_recommendations[key] = final_results[key]
                else:
                     # Handle case where a chain returns None or empty without exception
                     final_results[key] = {"error": f"Chain '{key}' returned no data."}
                     final_results["errors"].append(f"Chain '{key}' returned no data.")


            # --- Recommendations Chain ---
            if valid_analysis_for_recommendations: # Only run if at least one chain succeeded
                try:
                    logging.info("Generating final recommendations based on successful analyses.")
                    # Pass only the successful results to the recommendations prompt
                    recommendations_input = {"analysis_results": json.dumps(valid_analysis_for_recommendations, indent=2)}
                    recommendations_result = await self.recommendations_chain.ainvoke(recommendations_input) # Use await

                    # Check result type before dumping
                    if isinstance(recommendations_result, dict):
                         final_results["recommendations"] = recommendations_result
                    elif hasattr(recommendations_result, 'model_dump'):
                         final_results["recommendations"] = recommendations_result.model_dump()
                    else:
                         logging.error(f"Unexpected result type from recommendations_chain: {type(recommendations_result)}")
                         final_results["recommendations"] = {"error": "Recommendations chain returned unexpected data type."}
                         final_results["errors"].append("Recommendations chain returned unexpected data type.")

                except Exception as reco_exc:
                    error_msg = f"Error generating recommendations: {reco_exc}"
                    logging.error(error_msg, exc_info=True)
                    final_results["errors"].append(error_msg)
                    final_results["recommendations"] = {"error": f"Failed to generate recommendations: {reco_exc}"} # Use specific prefix
            else:
                 warning_msg = "Skipping recommendations as no analysis chains succeeded."
                 logging.warning(warning_msg)
                 final_results["errors"].append(warning_msg)
                 final_results["recommendations"] = {"error": warning_msg} # Store error info

        except Exception as e:
            # Catch any unexpected error during gather itself (less likely with return_exceptions=True)
            critical_error_msg = f"Critical error during enhance_seo_analysis gather: {e}"
            logging.error(critical_error_msg, exc_info=True)
            final_results["errors"].append(critical_error_msg)
            # Ensure basic structure exists even on critical failure, marking all as failed
            for key in analysis_keys:
                 final_results.setdefault(key, {"error": "Analysis failed due to critical error"})
            final_results.setdefault("recommendations", {"error": "Analysis failed due to critical error"})

        # Pass the potentially incomplete/error-containing results to formatting
        return self._format_output(final_results) # Ensure this return is correctly indented


    def _format_output(self, analysis_results: Dict) -> Dict: # Ensure this def is correctly indented at class level
        """Formats the analysis results, handling potential errors in sub-dictionaries."""
        # Ensure all code below is correctly indented under this method
        output = {
            "summary": {},
            "detailed_analysis": {},
            "quick_wins": [],
            "strategic_recommendations": [],
            "errors": analysis_results.get("errors", []) # Include overall errors list
        }

        # --- Process Detailed Analysis ---
        analysis_keys = [
            "entity_analysis", "credibility_analysis",
            "conversation_analysis", "cross_platform_presence",
            "recommendations"
        ]
        for key in analysis_keys:
            result_data = analysis_results.get(key)
            if isinstance(result_data, dict):
                output["detailed_analysis"][key] = result_data # Include the full dict (with potential 'error' key)
            else:
                # Handle cases where a key might be missing entirely (e.g., critical failure)
                output["detailed_analysis"][key] = {"error": f"Analysis data for '{key}' missing or invalid."}
                # Add error to main list if not already present via specific chain failure
                error_msg = f"Analysis data for '{key}' missing or invalid."
                if error_msg not in output["errors"]:
                     output["errors"].append(error_msg)


        # --- Calculate Scores (Safely) ---
        entity_data = output["detailed_analysis"].get("entity_analysis", {})
        cred_data = output["detailed_analysis"].get("credibility_analysis", {})
        conv_data = output["detailed_analysis"].get("conversation_analysis", {})
        plat_data = output["detailed_analysis"].get("cross_platform_presence", {})
        reco_data = output["detailed_analysis"].get("recommendations", {})

        # Use "Error" string if the sub-dict contains an 'error' key
        output["summary"]["entity_score"] = entity_data.get("knowledge_panel_readiness", "N/A") if "error" not in entity_data else "Error"

        credibility_score = "Error" # Default to Error if sub-dict has error
        if "error" not in cred_data:
            neeat_scores = cred_data.get("neeat_scores", {})
            if isinstance(neeat_scores, dict) and neeat_scores:
                valid_scores = [v for v in neeat_scores.values() if isinstance(v, (int, float))]
                if valid_scores:
                    # Calculate average only if valid_scores is not empty
                    credibility_score = sum(valid_scores) / len(valid_scores) if valid_scores else "N/A (No valid scores)"
                else: credibility_score = "N/A (No valid scores)" # Handle case of empty/invalid scores dict
            else: credibility_score = "N/A" # Handle case where neeat_scores is missing/not dict
        output["summary"]["credibility_score"] = credibility_score


        output["summary"]["conversation_score"] = conv_data.get("engagement_score", "N/A") if "error" not in conv_data else "Error"

        platform_score = "Error" # Default to Error if sub-dict has error
        if "error" not in plat_data:
            visibility_scores = plat_data.get("visibility_scores", {})
            if isinstance(visibility_scores, dict) and visibility_scores:
                valid_platform_scores = [v for v in visibility_scores.values() if isinstance(v, (int, float))]
                if valid_platform_scores:
                    # Calculate average only if valid_platform_scores is not empty
                    platform_score = sum(valid_platform_scores) / len(valid_platform_scores) if valid_platform_scores else "N/A (No valid scores)"
                else: platform_score = "N/A (No valid scores)"
            else: platform_score = "N/A"
        output["summary"]["platform_score"] = platform_score

        # --- Extract Top-Level Recommendations (Safely) ---
        if "error" not in reco_data:
            output["quick_wins"] = reco_data.get("quick_wins", [])
            output["strategic_recommendations"] = reco_data.get("strategic_recommendations", [])
        else:
            # If recommendations failed, populate lists with error message
            error_msg = f"Error retrieving recommendations: {reco_data.get('error', 'Unknown error')}"
            output["quick_wins"] = [error_msg]
            output["strategic_recommendations"] = [error_msg]
            # The specific error is already in the main errors list from enhance_seo_analysis


        return output


# Example usage with async support
async def enhanced_modern_analyze(
    site: str, sitemap: Optional[str] = None, api_key: str = None, **kwargs
):
    """
    Enhanced analysis incorporating modern SEO principles using LangChain
    """
    # Note: This example function might need adjustment depending on how analyze is called
    # It assumes 'analyze' returns the standard HTML-based analysis dictionary
    from pyseoanalyzer.analyzer import analyze # Corrected import path

    # Run original analysis (assuming standard mode, not fallback)
    # analyze is now async
    original_results = await analyze(site, sitemap, run_llm_analysis=False, **kwargs) # Ensure LLM isn't run twice

    # Enhance with modern SEO analysis if API key provided (or other LLM config is set)
    # We might want to check env vars directly here instead of relying on api_key
    llm_provider = os.environ.get("LLM_PROVIDER")
    if llm_provider: # Check if any LLM provider is configured
        logging.info("LLM provider configured, attempting enhancement.")
        enhancer = LLMSEOEnhancer()
        # Pass the results from the standard HTML analysis
        enhanced_results = await enhancer.enhance_seo_analysis(original_results)
        # The _format_output is now called within enhance_seo_analysis
        return enhanced_results
    else:
        logging.info("No LLM provider configured, returning original analysis results.")
        return original_results
