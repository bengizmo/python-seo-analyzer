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
        description="Individual N-E-E-A-T-U component scores (or nested structure)",
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

        llm_provider = os.environ.get("LLM_PROVIDER", "anthropic").lower()
        llm_model = os.environ.get("LLM_MODEL") # Specific model name
        logging.debug(f"Read LLM_MODEL from env: {llm_model}") # Add debug log
        temperature = float(os.environ.get("LLM_TEMPERATURE", 0.0))
        timeout = int(os.environ.get("LLM_TIMEOUT", 60)) # Increased timeout
        max_retries = int(os.environ.get("LLM_MAX_RETRIES", 3))

        logging.info(f"Initializing LLM Enhancer with provider: {llm_provider}")

        if llm_provider == "anthropic":
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
        elif llm_provider == "litellm":
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
                # Add other necessary LiteLLM parameters if needed
            )
            logging.info(f"Using LiteLLM model: {llm_model} via base: {api_base}")
        elif llm_provider == "ollama":
            base_url = os.environ.get("OLLAMA_BASE_URL")
            if not base_url:
                logging.error("OLLAMA_BASE_URL not found in environment variables.")
                raise ValueError("OLLAMA_BASE_URL is required for Ollama provider")
            if not llm_model:
                logging.error("LLM_MODEL must be specified for Ollama provider (e.g., 'mistral').")
                raise ValueError("LLM_MODEL is required for Ollama provider")
            self.llm = ChatOllama(
                base_url=base_url,
                model=llm_model,
                temperature=temperature,
                request_timeout=120, # Increased timeout to 120 seconds
                format="json", # Enforce JSON output format
                # Add other necessary Ollama parameters if needed
            )
            logging.info(f"Using Ollama model: {llm_model} via base: {base_url} with format=json") # Updated log
        else:
            logging.error(f"Unsupported LLM_PROVIDER: {llm_provider}. Defaulting to Anthropic.")
            # Defaulting to Anthropic as a fallback
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                logging.error("ANTHROPIC_API_KEY not found in environment variables for fallback.")
                raise ValueError("ANTHROPIC_API_KEY is required for default Anthropic provider")
            llm_model = "claude-3-sonnet-20240229" # Default Anthropic model
            self.llm = ChatAnthropic(
                model=llm_model,
                anthropic_api_key=api_key,
                temperature=temperature,
                timeout=timeout,
                max_retries=max_retries,
            )
            logging.info(f"Defaulted to Anthropic model: {llm_model}")

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

            IMPORTANT: Your response MUST be ONLY the JSON object itself, with NO additional text, explanations, or markdown formatting like ```json before or after the JSON object. Ensure all required fields (entity_assessment, knowledge_panel_readiness, key_improvements) are present.
            """
        )

        self.entity_chain = (
            {
                "seo_data": RunnablePassthrough(),
                "format_instructions": lambda _: entity_parser.get_format_instructions(),
            }
            | entity_prompt
            | self.llm
            | entity_parser
        )

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
              "entity_assessment": "The entity 'Example HVAC Co' shows moderate optimization. Clearer branding and structured data would improve Knowledge Panel readiness.",
              "knowledge_panel_readiness": 55,
              "key_improvements": [
                "Implement Organization schema markup.",
                "Ensure consistent NAP across citations.",
                "Develop a comprehensive 'About Us' page."
              ]
            }}
            ```

            IMPORTANT: Your response MUST be ONLY the JSON object itself, with NO additional text, explanations, or markdown formatting like ```json before or after the JSON object. Ensure all required fields (credibility_assessment, neeat_scores, trust_signals) are present.
            """
        )

        self.credibility_chain = (
            {
                "seo_data": RunnablePassthrough(),
                "format_instructions": lambda _: credibility_parser.get_format_instructions(),
            }
            | credibility_prompt
            | self.llm
            | credibility_parser
        )

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

            IMPORTANT: Your response MUST be ONLY the JSON object itself, with NO additional text, explanations, or markdown formatting like ```json before or after the JSON object. Ensure all required fields (conversation_readiness, query_patterns, engagement_score, gaps) are present. Pay close attention to the field name 'gaps'.
            """
        )

        self.conversation_chain = (
            {
                "seo_data": RunnablePassthrough(),
                "format_instructions": lambda _: conversation_parser.get_format_instructions(),
            }
            | conversation_prompt
            | self.llm
            | conversation_parser
        )

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

        self.platform_chain = (
            {
                "seo_data": RunnablePassthrough(),
                "format_instructions": lambda _: platform_parser.get_format_instructions(),
            }
            | platform_prompt
            | self.llm
            | platform_parser
        )

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

        self.recommendations_chain = (
            {
                "analysis_results": RunnablePassthrough(),
                "format_instructions": lambda _: recommendations_parser.get_format_instructions(),
            }
            | recommendations_prompt
            | self.llm
            | recommendations_parser
        )

    async def enhance_seo_analysis(self, seo_data: Dict) -> Dict:
        """
        Enhanced SEO analysis using modern LangChain patterns
        """
        # Convert seo_data to string for prompt insertion
        seo_data_str = json.dumps(seo_data, indent=2)

        # Run analysis chains in parallel
        entity_results, credibility_results, conversation_results, platform_results = (
            await asyncio.gather(
                self.entity_chain.ainvoke(seo_data_str),
                self.credibility_chain.ainvoke(seo_data_str),
                self.conversation_chain.ainvoke(seo_data_str),
                self.platform_chain.ainvoke(seo_data_str),
            )
        )

        # Combine analyses
        combined_analysis = {
            "entity_analysis": entity_results.model_dump(),
            "credibility_analysis": credibility_results.model_dump(),
            "conversation_analysis": conversation_results.model_dump(),
            "cross_platform_presence": platform_results.model_dump(),
        }

        # Generate final recommendations
        recommendations = await self.recommendations_chain.ainvoke(
            json.dumps(combined_analysis, indent=2)
        )

        # Combine all results
        final_results = {
            **seo_data,
            **combined_analysis,
            "recommendations": recommendations.model_dump(),
        }

        return self._format_output(final_results)

    def _format_output(self, raw_analysis: Dict) -> Dict:
        """Format analysis results into a clean, structured output"""
        return {
            "summary": {
                "entity_score": raw_analysis["entity_analysis"][
                    "knowledge_panel_readiness"
                ],
                "credibility_score": sum(
                    raw_analysis["credibility_analysis"]["neeat_scores"].values()
                )
                / 6,
                "conversation_score": raw_analysis["conversation_analysis"][
                    "engagement_score"
                ],
                "platform_score": sum(
                    raw_analysis["cross_platform_presence"][
                        "visibility_scores"
                    ].values()
                )
                / len(raw_analysis["cross_platform_presence"]["visibility_scores"]),
            },
            "detailed_analysis": raw_analysis,
            "quick_wins": raw_analysis["recommendations"]["quick_wins"],
            "strategic_recommendations": raw_analysis["recommendations"][
                "strategic_recommendations"
            ],
        }


# Example usage with async support
async def enhanced_modern_analyze(
    site: str, sitemap: Optional[str] = None, api_key: str = None, **kwargs
):
    """
    Enhanced analysis incorporating modern SEO principles using LangChain
    """
    from pyseoanalyzer import analyze

    # Run original analysis
    original_results = analyze(site, sitemap, **kwargs)

    # Enhance with modern SEO analysis if API key provided
    if api_key:
        enhancer = LLMSEOEnhancer()
        enhanced_results = await enhancer.enhance_seo_analysis(original_results)
        return enhancer._format_output(enhanced_results)

    return original_results
