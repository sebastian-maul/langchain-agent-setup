import mlflow
import requests
from typing import Optional, Dict, Any


class MLflowLoggingSettings:
    """Configuration and utilities for MLflow logging"""
    
    def __init__(
        self,
        tracking_uri: str = "http://127.0.0.1:5000",
        experiment_name: str = "first steps",
        enable_system_metrics: bool = True,
        enable_langchain_autolog: bool = True
    ):
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.enable_system_metrics = enable_system_metrics
        self.enable_langchain_autolog = enable_langchain_autolog
    
    def setup_mlflow(self) -> None:
        """Initialize MLflow with the configured settings"""
        mlflow.set_tracking_uri(uri=self.tracking_uri)
        
        if self.enable_system_metrics:
            mlflow.enable_system_metrics_logging()
        
        if self.enable_langchain_autolog:
            mlflow.langchain.autolog()
        
        mlflow.set_experiment(self.experiment_name)
        mlflow.start_run()
    
    def get_ollama_model_info(self, model_name: str, ollama_url: str = "http://localhost:11434") -> Optional[Dict[str, Any]]:
        """Get model information from Ollama API"""
        try:
            response = requests.post(
                f"{ollama_url}/api/show",
                json={"model": model_name},
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Warning: Could not fetch model info for {model_name}: {e}")
            return None
    
    def log_ollama_model_metadata(self, model_name: str, model_info: Optional[Dict[str, Any]]) -> None:
        """Log Ollama model metadata to MLflow"""
        if not model_info:
            return
        
        try:
            # Log basic model info
            mlflow.log_param("ollama_model", model_name)
            
            # Log parameter information from details
            if "details" in model_info:
                details = model_info["details"]
                if "parameter_size" in details:
                    mlflow.log_param("parameter_size", details["parameter_size"])
                if "quantization_level" in details:
                    mlflow.log_param("quantization_level", details["quantization_level"])
                if "format" in details:
                    mlflow.log_param("model_format", details["format"])
                if "family" in details:
                    mlflow.log_param("model_family", details["family"])
            
            # Log model_info details (this contains the metadata keys we saw)
            if "model_info" in model_info:
                model_info_dict = model_info["model_info"]
                
                # Architecture
                if "general.architecture" in model_info_dict:
                    mlflow.log_param("architecture", model_info_dict["general.architecture"])
                
                # Base model information
                if "general.base_model.0.name" in model_info_dict:
                    mlflow.log_param("base_model_name", model_info_dict["general.base_model.0.name"])
                
                if "general.base_model.0.organization" in model_info_dict:
                    mlflow.log_param("base_model_org", model_info_dict["general.base_model.0.organization"])
                
                if "general.base_model.0.version" in model_info_dict:
                    mlflow.log_param("base_model_version", model_info_dict["general.base_model.0.version"])
                
                # Context length (try different possible keys)
                context_keys = ["llama.context_length", "context_length", "general.context_length"]
                for key in context_keys:
                    if key in model_info_dict:
                        mlflow.log_param("context_length", model_info_dict[key])
                        break
                
                # Parameter count (try different possible keys)
                param_keys = ["general.parameter_count", "parameter_count"]
                for key in param_keys:
                    if key in model_info_dict:
                        mlflow.log_param("parameter_count", model_info_dict[key])
                        break
            
            # Log modification date
            if "modified_at" in model_info:
                mlflow.log_param("model_modified_at", model_info["modified_at"])
            
            # Log full model info as artifact for reference
            mlflow.log_dict(model_info, "model_info.json")
            
            print(f"✓ Successfully logged metadata for model: {model_name}")
            
        except Exception as e:
            print(f"Warning: Could not log model metadata: {e}")
    
    def log_model_and_metadata(self, model_name: str, ollama_url: str = "http://localhost:11434") -> None:
        """Get model info from Ollama and log metadata to MLflow"""
        model_info = self.get_ollama_model_info(model_name, ollama_url)
        self.log_ollama_model_metadata(model_name, model_info)
    
    def print_token_usage_summary(self, traces: list) -> None:
        """Print a summary of token usage from MLflow traces"""
        total_in = 0
        total_out = 0
        total_usage = 0

        # Print the token usage
        for trace in traces:
            if hasattr(trace.info, 'token_usage') and trace.info.token_usage:
                total_in += trace.info.token_usage.get('input_tokens', 0)
                total_out += trace.info.token_usage.get('output_tokens', 0)
                total_usage += trace.info.token_usage.get('total_tokens', 0)

        print("== Total token usage: ==")
        print(f"  Input tokens: {total_in}")
        print(f"  Output tokens: {total_out}")
        print(f"  Total tokens: {total_usage}")
