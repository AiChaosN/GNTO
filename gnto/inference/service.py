"""
Inference service for production deployment
"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
import time
import json
from pathlib import Path

from ..core.model import LQOModel
from ..core.feature_spec import FeatureSpec


logger = logging.getLogger(__name__)


class InferenceService:
    """
    Production inference service for the LQO model
    
    Provides high-level interfaces for plan prediction, fallback detection,
    and performance monitoring in production environments.
    """
    
    def __init__(
        self,
        model: LQOModel,
        fallback_threshold: Dict[str, float] = None,
        enable_monitoring: bool = True,
        batch_size: int = 32,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.model.eval()  # Set to evaluation mode
        
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Fallback thresholds for uncertainty-based fallback
        self.fallback_threshold = fallback_threshold or {
            "uncertainty_threshold": 0.5,
            "cost_confidence_threshold": 0.8,
            "latency_confidence_threshold": 0.8,
            "min_plan_score": 0.1
        }
        
        self.enable_monitoring = enable_monitoring
        self.batch_size = batch_size
        
        # Monitoring statistics
        self.stats = {
            "total_predictions": 0,
            "fallback_count": 0,
            "avg_inference_time": 0.0,
            "error_count": 0,
            "last_reset": time.time()
        }
        
        logger.info(f"Initialized InferenceService on device: {self.device}")
        logger.info(f"Fallback thresholds: {self.fallback_threshold}")
    
    def predict(self, plan_json: Union[Dict[str, Any], List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Main prediction interface
        
        Args:
            plan_json: Single execution plan or list of plans
        
        Returns:
            Dictionary containing predictions and metadata
        """
        start_time = time.time()
        
        try:
            # Ensure input is a list
            if isinstance(plan_json, dict):
                plans = [plan_json]
                single_plan = True
            else:
                plans = plan_json
                single_plan = False
            
            # Batch processing for efficiency
            all_predictions = []
            
            for i in range(0, len(plans), self.batch_size):
                batch_plans = plans[i:i + self.batch_size]
                batch_predictions = self._predict_batch(batch_plans)
                all_predictions.extend(batch_predictions)
            
            # Return single prediction if single plan input
            if single_plan:
                result = all_predictions[0]
            else:
                result = {
                    "predictions": all_predictions,
                    "batch_size": len(plans)
                }
            
            # Add timing information
            inference_time = time.time() - start_time
            result["inference_time"] = inference_time
            
            # Update monitoring stats
            if self.enable_monitoring:
                self._update_stats(inference_time, success=True)
            
            return result
        
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            
            if self.enable_monitoring:
                self._update_stats(time.time() - start_time, success=False)
            
            return {
                "error": str(e),
                "inference_time": time.time() - start_time,
                "fallback_recommended": True
            }
    
    def _predict_batch(self, batch_plans: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a batch of plans"""
        with torch.no_grad():
            # Get model predictions
            predictions = self.model.predict(batch_plans)
            
            # Convert to list of individual predictions
            batch_size = len(batch_plans)
            individual_predictions = []
            
            for i in range(batch_size):
                plan_pred = {}
                
                # Extract predictions for this plan
                for key, values in predictions.items():
                    if isinstance(values, torch.Tensor):
                        if values.dim() == 1:
                            plan_pred[key] = float(values[i])
                        elif values.dim() == 2:
                            plan_pred[key] = values[i].tolist()
                        else:
                            plan_pred[key] = values[i]
                    else:
                        plan_pred[key] = values
                
                # Add confidence scores and fallback recommendation
                confidence_info = self._compute_confidence(plan_pred)
                plan_pred.update(confidence_info)
                
                individual_predictions.append(plan_pred)
            
            return individual_predictions
    
    def _compute_confidence(self, prediction: Dict[str, Any]) -> Dict[str, Any]:
        """Compute confidence scores and fallback recommendation"""
        confidence_info = {
            "confidence_scores": {},
            "overall_confidence": 1.0,
            "fallback_recommended": False,
            "fallback_reasons": []
        }
        
        # Task-specific confidence computation
        if "uncertainty" in prediction:
            uncertainty = prediction["uncertainty"]
            if isinstance(uncertainty, dict):
                # Handle different uncertainty types
                if "aleatoric_var" in uncertainty:
                    aleatoric_uncertainty = float(torch.tensor(uncertainty["aleatoric_var"]).mean())
                    confidence_info["confidence_scores"]["aleatoric"] = max(0.0, 1.0 - aleatoric_uncertainty)
                
                if "epistemic_var" in uncertainty:
                    epistemic_uncertainty = float(torch.tensor(uncertainty["epistemic_var"]).mean())
                    confidence_info["confidence_scores"]["epistemic"] = max(0.0, 1.0 - epistemic_uncertainty)
        
        # Cost/latency prediction confidence
        for task in ["cost", "latency", "memory", "mem"]:
            if task in prediction:
                # Simple heuristic: higher values might be less reliable
                pred_value = prediction[task]
                if isinstance(pred_value, (list, tuple)):
                    pred_value = pred_value[0] if pred_value else 0.0
                
                # Normalize confidence based on prediction magnitude
                confidence = 1.0 / (1.0 + abs(pred_value) * 0.01)  # Simple heuristic
                confidence_info["confidence_scores"][task] = confidence
        
        # Ranking confidence
        if "rank_scores" in prediction or "ranking" in prediction:
            rank_key = "rank_scores" if "rank_scores" in prediction else "ranking"
            rank_score = prediction[rank_key]
            if isinstance(rank_score, (list, tuple)):
                rank_score = rank_score[0] if rank_score else 0.0
            
            # Higher absolute ranking scores indicate more confidence
            confidence_info["confidence_scores"]["ranking"] = min(1.0, abs(rank_score))
        
        # Compute overall confidence
        if confidence_info["confidence_scores"]:
            overall_confidence = np.mean(list(confidence_info["confidence_scores"].values()))
            confidence_info["overall_confidence"] = overall_confidence
        
        # Determine fallback recommendation
        fallback_reasons = []
        
        # Check uncertainty threshold
        if confidence_info["overall_confidence"] < self.fallback_threshold["uncertainty_threshold"]:
            fallback_reasons.append("low_overall_confidence")
        
        # Check specific task confidence
        for task in ["cost", "latency"]:
            threshold_key = f"{task}_confidence_threshold"
            if (threshold_key in self.fallback_threshold and 
                task in confidence_info["confidence_scores"] and
                confidence_info["confidence_scores"][task] < self.fallback_threshold[threshold_key]):
                fallback_reasons.append(f"low_{task}_confidence")
        
        # Check minimum plan score
        if ("ranking" in confidence_info["confidence_scores"] and 
            confidence_info["confidence_scores"]["ranking"] < self.fallback_threshold["min_plan_score"]):
            fallback_reasons.append("low_plan_score")
        
        confidence_info["fallback_recommended"] = len(fallback_reasons) > 0
        confidence_info["fallback_reasons"] = fallback_reasons
        
        return confidence_info
    
    def should_fallback(self, prediction: Dict[str, Any]) -> bool:
        """
        Determine if the system should fallback to traditional optimizer
        
        Args:
            prediction: Prediction dictionary from predict()
        
        Returns:
            Boolean indicating whether to fallback
        """
        if "fallback_recommended" in prediction:
            return prediction["fallback_recommended"]
        
        # Fallback for error cases
        if "error" in prediction:
            return True
        
        # Additional fallback logic can be added here
        return False
    
    def batch_predict(
        self, 
        plan_jsons: List[Dict[str, Any]], 
        return_embeddings: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Batch prediction interface for multiple plans
        
        Args:
            plan_jsons: List of execution plans
            return_embeddings: Whether to include plan embeddings
        
        Returns:
            List of prediction dictionaries
        """
        predictions = []
        
        # Process in batches
        for i in range(0, len(plan_jsons), self.batch_size):
            batch_plans = plan_jsons[i:i + self.batch_size]
            
            # Get predictions
            batch_result = self.predict(batch_plans)
            
            if "predictions" in batch_result:
                batch_predictions = batch_result["predictions"]
            else:
                batch_predictions = [batch_result]
            
            # Add embeddings if requested
            if return_embeddings:
                try:
                    embeddings = self.model.get_plan_embedding(batch_plans)
                    for j, pred in enumerate(batch_predictions):
                        pred["embedding"] = embeddings[j].tolist()
                except Exception as e:
                    logger.warning(f"Failed to get embeddings: {e}")
            
            predictions.extend(batch_predictions)
        
        return predictions
    
    def get_stats(self) -> Dict[str, Any]:
        """Get monitoring statistics"""
        if not self.enable_monitoring:
            return {"monitoring_disabled": True}
        
        current_time = time.time()
        uptime = current_time - self.stats["last_reset"]
        
        stats = self.stats.copy()
        stats["uptime_seconds"] = uptime
        
        if stats["total_predictions"] > 0:
            stats["fallback_rate"] = stats["fallback_count"] / stats["total_predictions"]
            stats["error_rate"] = stats["error_count"] / stats["total_predictions"]
        else:
            stats["fallback_rate"] = 0.0
            stats["error_rate"] = 0.0
        
        return stats
    
    def reset_stats(self):
        """Reset monitoring statistics"""
        self.stats = {
            "total_predictions": 0,
            "fallback_count": 0,
            "avg_inference_time": 0.0,
            "error_count": 0,
            "last_reset": time.time()
        }
        logger.info("Reset monitoring statistics")
    
    def _update_stats(self, inference_time: float, success: bool):
        """Update monitoring statistics"""
        self.stats["total_predictions"] += 1
        
        if not success:
            self.stats["error_count"] += 1
        
        # Update average inference time (exponential moving average)
        alpha = 0.1
        if self.stats["avg_inference_time"] == 0.0:
            self.stats["avg_inference_time"] = inference_time
        else:
            self.stats["avg_inference_time"] = (
                alpha * inference_time + 
                (1 - alpha) * self.stats["avg_inference_time"]
            )
    
    def warmup(self, num_warmup: int = 10):
        """
        Warmup the model with dummy predictions to optimize performance
        
        Args:
            num_warmup: Number of warmup predictions
        """
        logger.info(f"Starting model warmup with {num_warmup} predictions...")
        
        # Create dummy plan for warmup
        dummy_plan = {
            "nodes": [
                {
                    "operator_type": "SeqScan",
                    "rows": 1000.0,
                    "ndv": 100.0,
                    "selectivity": 1.0,
                    "io_cost": 10.0,
                    "cpu_cost": 5.0,
                    "parallel_degree": 1.0,
                    "join_type": "none",
                    "index_type": "none",
                    "storage_format": "heap",
                    "hint": "none",
                    "is_blocking": 0.0,
                    "is_pipeline": 1.0,
                    "is_probe": 0.0,
                    "is_build": 0.0,
                    "stage_id": 0.0
                }
            ],
            "edges": []
        }
        
        start_time = time.time()
        
        for i in range(num_warmup):
            try:
                _ = self.predict(dummy_plan)
            except Exception as e:
                logger.warning(f"Warmup prediction {i} failed: {e}")
        
        warmup_time = time.time() - start_time
        logger.info(f"Warmup completed in {warmup_time:.2f}s")
    
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: Optional[torch.device] = None,
        **kwargs
    ) -> "InferenceService":
        """
        Create inference service from model checkpoint
        
        Args:
            checkpoint_path: Path to model checkpoint
            device: Device to load model on
            **kwargs: Additional arguments for InferenceService
        
        Returns:
            InferenceService instance
        """
        # Load model from checkpoint
        model = LQOModel.load_checkpoint(checkpoint_path, device=device)
        
        # Create inference service
        service = cls(model=model, device=device, **kwargs)
        
        logger.info(f"Created InferenceService from checkpoint: {checkpoint_path}")
        
        return service
