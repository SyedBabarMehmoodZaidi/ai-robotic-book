---
sidebar_position: 3
---

# Multimodal Perception: Vision-Language Integration

## Learning Objectives

By the end of this section, you will be able to:
- Understand multimodal perception architectures for vision-language fusion
- Implement visual grounding and object detection with language conditioning
- Create spatial reasoning systems that connect language to visual scenes
- Design multimodal feature extraction and fusion mechanisms
- Evaluate multimodal perception performance in robotic contexts
- Integrate multimodal perception with robotic action systems

## Introduction to Multimodal Perception

**Multimodal perception** in robotics involves the integration of multiple sensory modalities, primarily vision and language, to create a comprehensive understanding of the environment. Unlike traditional perception systems that process each modality independently, multimodal perception systems combine information from different sources to achieve more robust and semantically rich scene understanding.

Multimodal perception enables robots to:
- Understand natural language descriptions of visual scenes
- Ground language commands to specific visual objects and locations
- Perform spatial reasoning based on both visual and linguistic information
- Handle ambiguous or incomplete information through cross-modal inference
- Interact naturally with humans using both visual and linguistic cues

## Multimodal Perception Architecture

### Core Components

#### 1. Visual Processing Pipeline

```python
# visual_processing.py
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from transformers import CLIPVisionModel
import cv2
import numpy as np

class VisualProcessor(nn.Module):
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        super(VisualProcessor, self).__init__()

        # CLIP vision encoder
        self.clip_vision = CLIPVisionModel.from_pretrained(model_name)

        # Spatial feature extractor
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7))
        )

        # Feature fusion layer
        self.feature_fusion = nn.Sequential(
            nn.Linear(512 + 128, 512),  # CLIP + spatial features
            nn.LayerNorm(512),
            nn.ReLU()
        )

        # Spatial position encoding
        self.pos_encoder = nn.Parameter(torch.randn(1, 50, 512))  # 7x7 = 49 + 1 for global

    def forward(self, images):
        batch_size, channels, height, width = images.shape

        # Process with CLIP vision encoder
        clip_features = self.clip_vision(pixel_values=images)
        global_features = clip_features.pooler_output  # (batch, 512)
        patch_features = clip_features.last_hidden_state  # (batch, num_patches, 768)

        # Extract spatial features
        spatial_features = self.spatial_encoder(images)  # (batch, 128, 7, 7)
        spatial_features_flat = spatial_features.view(batch_size, 128, -1).transpose(-1, -2)  # (batch, 49, 128)

        # Fuse visual features
        # Repeat global features for each spatial location
        global_repeated = global_features.unsqueeze(1).repeat(1, 49, 1)  # (batch, 49, 512)

        # Concatenate and fuse
        fused_features = self.feature_fusion(
            torch.cat([global_repeated, spatial_features_flat], dim=-1)
        )

        # Add global feature
        global_feature_expanded = global_features.unsqueeze(1)  # (batch, 1, 512)
        all_features = torch.cat([global_feature_expanded, fused_features], dim=1)  # (batch, 50, 512)

        # Add positional encoding
        pos_enc = self.pos_encoder[:, :all_features.size(1), :].expand(batch_size, -1, -1)
        final_features = all_features + pos_enc

        return {
            'features': final_features,  # (batch, seq_len, dim)
            'global_features': global_features,  # (batch, dim)
            'spatial_features': fused_features  # (batch, 49, 512)
        }
```

#### 2. Language Processing Pipeline

```python
# language_processing.py
import torch
import torch.nn as nn
from transformers import LlamaModel, LlamaTokenizer
import re

class LanguageProcessor(nn.Module):
    def __init__(self, model_name="meta-llama/Llama-2-7b-hf"):
        super(LanguageProcessor, self).__init__()

        self.tokenizer = LlamaTokenizer.from_pretrained(model_name)
        self.llm = LlamaModel.from_pretrained(model_name)

        # Add special tokens for multimodal processing
        special_tokens = {
            'additional_special_tokens': [
                '<OBJECT>', '</OBJECT>',  # For object references
                '<LOCATION>', '</LOCATION>',  # For spatial references
                '<ACTION>', '</ACTION>',  # For action descriptions
                '<VISUAL>', '</VISUAL>'   # For visual context
            ]
        }
        self.tokenizer.add_special_tokens(special_tokens)

        # Projection layer for visual features
        self.visual_projection = nn.Linear(512, self.llm.config.hidden_size)

        # Attention mechanism for visual grounding
        self.visual_attention = nn.MultiheadAttention(
            embed_dim=self.llm.config.hidden_size,
            num_heads=8,
            batch_first=True
        )

        # Classification head for grounding
        self.grounding_head = nn.Linear(self.llm.config.hidden_size, 1)

    def tokenize_text(self, texts, max_length=512):
        """Tokenize text with special handling for multimodal content"""
        # Add visual context markers if not present
        processed_texts = []
        for text in texts:
            if '<VISUAL>' not in text:
                text = f'<VISUAL>{text}</VISUAL>'
            processed_texts.append(text)

        return self.tokenizer(
            processed_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )

    def forward(self, input_ids, attention_mask, visual_features=None):
        # Process language tokens
        language_outputs = self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        language_features = language_outputs.last_hidden_state  # (batch, seq_len, hidden_size)

        if visual_features is not None:
            # Apply cross-attention with visual features
            visual_projected = self.visual_projection(visual_features)  # (batch, vis_seq_len, hidden_size)

            # Apply attention: language attends to visual features
            attended_features, attention_weights = self.visual_attention(
                language_features,  # query
                visual_projected,   # key
                visual_projected    # value
            )

            # Combine original language features with attended visual features
            combined_features = language_features + attended_features

            # Get grounding scores for each token
            grounding_scores = self.grounding_head(combined_features)  # (batch, seq_len, 1)

            return {
                'features': combined_features,
                'attention_weights': attention_weights,
                'grounding_scores': grounding_scores,
                'language_features': language_features
            }

        return {
            'features': language_features,
            'language_features': language_features
        }

    def extract_entities(self, text, grounding_scores, threshold=0.5):
        """Extract entities and their grounding scores from text"""
        tokens = self.tokenizer.tokenize(text)

        entities = []
        current_entity = ""
        current_score = 0
        entity_count = 0

        for i, (token, score) in enumerate(zip(tokens, grounding_scores[0])):
            if score > threshold:
                if current_entity:
                    current_entity += " " + token
                else:
                    current_entity = token
                    entity_start = i
                current_score += score.item()
            else:
                if current_entity:
                    entities.append({
                        'text': current_entity,
                        'score': current_score / len(current_entity.split()),
                        'start_idx': entity_start,
                        'end_idx': i
                    })
                    current_entity = ""
                    current_score = 0

        if current_entity:
            entities.append({
                'text': current_entity,
                'score': current_score / len(current_entity.split()),
                'start_idx': entity_start,
                'end_idx': len(tokens)
            })

        return entities
```

#### 3. Multimodal Fusion Mechanism

```python
# multimodal_fusion.py
import torch
import torch.nn as nn

class MultimodalFusion(nn.Module):
    def __init__(self, visual_dim=512, language_dim=4096, output_dim=512):
        super(MultimodalFusion, self).__init__()

        self.visual_dim = visual_dim
        self.language_dim = language_dim
        self.output_dim = output_dim

        # Projection layers to match dimensions
        self.visual_project = nn.Linear(visual_dim, output_dim)
        self.language_project = nn.Linear(language_dim, output_dim)

        # Cross-attention mechanism
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=8,
            batch_first=True
        )

        # Self-attention for multimodal features
        self.self_attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=8,
            batch_first=True
        )

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(output_dim, output_dim * 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim * 4, output_dim)
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(output_dim)
        self.norm2 = nn.LayerNorm(output_dim)

        # Final output projection
        self.output_projection = nn.Linear(output_dim, output_dim)

    def forward(self, visual_features, language_features):
        """
        Fuse visual and language features

        Args:
            visual_features: (batch, vis_seq_len, visual_dim)
            language_features: (batch, lang_seq_len, language_dim)
        """
        batch_size = visual_features.size(0)

        # Project features to common dimension
        vis_proj = self.visual_project(visual_features)  # (batch, vis_seq_len, output_dim)
        lang_proj = self.language_project(language_features)  # (batch, lang_seq_len, output_dim)

        # Cross-attention: visual features attend to language features
        vis_attended, vis_attention = self.cross_attention(
            vis_proj,  # query
            lang_proj,  # key
            lang_proj   # value
        )

        # Cross-attention: language features attend to visual features
        lang_attended, lang_attention = self.cross_attention(
            lang_proj,  # query
            vis_proj,   # key
            vis_proj    # value
        )

        # Concatenate attended features
        combined_features = torch.cat([
            vis_attended,
            lang_attended
        ], dim=1)  # (batch, vis_seq_len + lang_seq_len, output_dim)

        # Self-attention on combined features
        attended_combined, self_attention = self.self_attention(
            combined_features,
            combined_features,
            combined_features
        )

        # Add & Norm
        norm1_output = self.norm1(combined_features + attended_combined)

        # Feed Forward & Norm
        ffn_output = self.ffn(norm1_output)
        multimodal_features = self.norm2(norm1_output + ffn_output)

        # Final projection
        final_features = self.output_projection(multimodal_features)

        return {
            'fused_features': final_features,
            'vis_attention': vis_attention,
            'lang_attention': lang_attention,
            'self_attention': self_attention
        }
```

## Visual Grounding Systems

### Object Detection with Language Conditioning

```python
# visual_grounding.py
import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from transformers import LlamaModel

class VisualGroundingSystem(nn.Module):
    def __init__(self, language_model_name="meta-llama/Llama-2-7b-hf"):
        super(VisualGroundingSystem, self).__init__()

        # Object detection backbone
        self.object_detector = fasterrcnn_resnet50_fpn(pretrained=True)
        detector_features = self.object_detector.backbone.out_channels

        # Language encoder
        self.language_encoder = LlamaModel.from_pretrained(language_model_name)
        self.lang_proj = nn.Linear(self.language_encoder.config.hidden_size, 256)

        # Visual feature projection
        self.vis_proj = nn.Linear(detector_features, 256)

        # Cross-modal attention for grounding
        self.grounding_attention = nn.MultiheadAttention(
            embed_dim=256,
            num_heads=8,
            batch_first=True
        )

        # Grounding score predictor
        self.grounding_predictor = nn.Linear(256, 1)

        # Bounding box refinement
        self.bbox_refinement = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 4)  # dx, dy, dw, dh
        )

    def forward(self, images, text_queries):
        """
        Ground text queries to visual objects

        Args:
            images: List of images or batched tensor
            text_queries: List of text queries for each image
        """
        batch_size = len(images) if isinstance(images, list) else images.size(0)

        # Extract visual features using object detector
        if isinstance(images, list):
            # Convert list to tensor if needed
            images_tensor = torch.stack([img for img in images])
        else:
            images_tensor = images

        # Get visual features from backbone
        visual_features = self.object_detector.backbone(images_tensor)

        # Extract object proposals
        proposals = self.object_detector.rpn(images_tensor, visual_features)

        # Get object detections
        detections = self.object_detector.roi_heads(visual_features, proposals, images_tensor.shape[-2:])

        # Process language queries
        tokenized_queries = []
        for query in text_queries:
            tokens = self.language_encoder.embed_tokens(
                torch.tensor([self.language_encoder.config.vocab_size-1])  # Simplified
            )
            tokenized_queries.append(tokens)

        # In practice, you would properly tokenize and encode the text queries
        # For this example, we'll use a simplified approach

        # Project and fuse features
        # This is a simplified version - in practice, you would:
        # 1. Extract RoI features for detected objects
        # 2. Encode text queries
        # 3. Apply cross-attention between visual objects and text
        # 4. Compute grounding scores

        # Placeholder for actual grounding computation
        grounding_scores = torch.rand(batch_size, len(detections), 1)  # (batch, num_objects, 1)

        results = []
        for i in range(batch_size):
            obj_detections = detections[i]
            scores = grounding_scores[i].squeeze(-1)

            # Sort by grounding confidence
            sorted_indices = torch.argsort(scores, descending=True)

            # Apply NMS based on grounding scores
            selected_indices = torchvision.ops.nms(
                obj_detections['boxes'][sorted_indices],
                scores[sorted_indices],
                iou_threshold=0.5
            )

            selected_boxes = obj_detections['boxes'][sorted_indices][selected_indices]
            selected_scores = scores[sorted_indices][selected_indices]

            results.append({
                'boxes': selected_boxes,
                'grounding_scores': selected_scores,
                'labels': obj_detections.get('labels', torch.zeros(len(selected_boxes)))
            })

        return results

class AdvancedVisualGrounding(nn.Module):
    def __init__(self, vision_model_name="openai/clip-vit-base-patch32"):
        super(AdvancedVisualGrounding, self).__init__()

        # Use CLIP for visual grounding
        from transformers import CLIPProcessor, CLIPModel
        self.clip_model = CLIPModel.from_pretrained(vision_model_name)
        self.clip_processor = CLIPProcessor.from_pretrained(vision_model_name)

        # Additional grounding head
        self.grounding_head = nn.Linear(512, 1)  # 512 is CLIP's feature dimension

    def forward(self, pixel_values, text_descriptions):
        """
        Perform visual grounding using CLIP

        Args:
            pixel_values: Image tensor
            text_descriptions: List of text descriptions
        """
        # Get image and text features from CLIP
        image_features = self.clip_model.get_image_features(pixel_values=pixel_values)

        # Process text descriptions
        text_inputs = self.clip_processor.tokenizer(
            text_descriptions,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        text_features = self.clip_model.get_text_features(**text_inputs)

        # Compute similarity scores
        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Cosine similarity
        similarity_scores = torch.matmul(image_features, text_features.T)

        return similarity_scores
```

### Spatial Reasoning and Layout Understanding

```python
# spatial_reasoning.py
import torch
import torch.nn as nn
import numpy as np

class SpatialReasoningModule(nn.Module):
    def __init__(self, feature_dim=512):
        super(SpatialReasoningModule, self).__init__()

        self.feature_dim = feature_dim

        # Spatial relationship encoder
        self.spatial_encoder = nn.Sequential(
            nn.Linear(4, 128),  # 4 coordinates (x1, y1, x2, y2)
            nn.ReLU(),
            nn.Linear(128, feature_dim)
        )

        # Relative position encoder
        self.relative_position_encoder = nn.Sequential(
            nn.Linear(2, 64),  # 2D relative position (dx, dy)
            nn.ReLU(),
            nn.Linear(64, feature_dim)
        )

        # Spatial relation classifier
        self.relation_classifier = nn.Sequential(
            nn.Linear(feature_dim * 3, 256),  # obj1 + obj2 + spatial
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 6)  # 6 common spatial relations: left, right, above, below, near, far
        )

        # Spatial attention mechanism
        self.spatial_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=8,
            batch_first=True
        )

    def compute_spatial_features(self, boxes):
        """
        Compute spatial features from bounding boxes

        Args:
            boxes: Tensor of shape (batch, num_objects, 4) [x1, y1, x2, y2]
        """
        # Compute geometric properties
        x1, y1, x2, y2 = boxes.unbind(-1)

        # Basic spatial features
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        width = x2 - x1
        height = y2 - y1
        area = width * height

        # Normalize coordinates (assuming image size normalization)
        # In practice, you would normalize by image dimensions
        spatial_features = torch.stack([center_x, center_y, width, height], dim=-1)

        return spatial_features

    def compute_relative_positions(self, boxes):
        """
        Compute relative positions between all pairs of objects
        """
        batch_size, num_objects, _ = boxes.shape

        # Expand boxes for pairwise comparison
        boxes_expanded_1 = boxes.unsqueeze(2).expand(-1, -1, num_objects, -1)  # (B, N, N, 4)
        boxes_expanded_2 = boxes.unsqueeze(1).expand(-1, num_objects, -1, -1)  # (B, N, N, 4)

        # Compute relative centers
        center1_x = (boxes_expanded_1[..., 0] + boxes_expanded_1[..., 2]) / 2
        center1_y = (boxes_expanded_1[..., 1] + boxes_expanded_1[..., 3]) / 2
        center2_x = (boxes_expanded_2[..., 0] + boxes_expanded_2[..., 2]) / 2
        center2_y = (boxes_expanded_2[..., 1] + boxes_expanded_2[..., 3]) / 2

        # Compute relative positions
        rel_x = center2_x - center1_x  # (B, N, N)
        rel_y = center2_y - center1_y  # (B, N, N)

        return torch.stack([rel_x, rel_y], dim=-1)  # (B, N, N, 2)

    def forward(self, visual_features, boxes):
        """
        Perform spatial reasoning on visual features and bounding boxes

        Args:
            visual_features: (batch, num_objects, feature_dim)
            boxes: (batch, num_objects, 4) [x1, y1, x2, y2]
        """
        batch_size, num_objects, _ = visual_features.shape

        # Compute spatial features
        spatial_features = self.compute_spatial_features(boxes)  # (batch, num_objects, 4)
        spatial_embeddings = self.spatial_encoder(spatial_features)  # (batch, num_objects, feature_dim)

        # Compute relative positions
        relative_positions = self.compute_relative_positions(boxes)  # (batch, num_objects, num_objects, 2)

        # Encode relative positions
        rel_pos_embeddings = self.relative_position_encoder(relative_positions)  # (batch, num_objects, num_objects, feature_dim)

        # Combine visual and spatial features
        combined_features = visual_features + spatial_embeddings  # (batch, num_objects, feature_dim)

        # Apply spatial attention
        attended_features, attention_weights = self.spatial_attention(
            combined_features,  # query
            combined_features,  # key
            combined_features   # value
        )

        # Compute spatial relations between all pairs
        relation_features = []
        for i in range(num_objects):
            for j in range(num_objects):
                if i != j:
                    # Combine features of object i and j with spatial relation
                    obj_pair_features = torch.cat([
                        attended_features[:, i, :],  # Feature of object i
                        attended_features[:, j, :],  # Feature of object j
                        rel_pos_embeddings[:, i, j, :]  # Spatial relation embedding
                    ], dim=-1)  # (batch, feature_dim * 3)

                    relation_features.append(obj_pair_features)

        # If there are pairs, classify their spatial relations
        if relation_features:
            all_relation_features = torch.stack(relation_features, dim=1)  # (batch, num_pairs, feature_dim * 3)
            spatial_relations = self.relation_classifier(all_relation_features)  # (batch, num_pairs, 6)
        else:
            spatial_relations = torch.zeros(batch_size, 0, 6)

        return {
            'attended_features': attended_features,
            'spatial_relations': spatial_relations,
            'attention_weights': attention_weights,
            'spatial_features': spatial_embeddings
        }
```

## Scene Understanding with Language

### Semantic Scene Graph Generation

```python
# scene_graph.py
import torch
import torch.nn as nn

class SceneGraphGenerator(nn.Module):
    def __init__(self, vocab_size=10000, feature_dim=512):
        super(SceneGraphGenerator, self).__init__()

        self.feature_dim = feature_dim

        # Object detection and classification
        self.object_classifier = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, vocab_size)  # Number of object categories
        )

        # Relationship classifier
        self.relation_classifier = nn.Sequential(
            nn.Linear(feature_dim * 2 + 4, 512),  # 2 obj features + 4 spatial coords
            nn.ReLU(),
            nn.Linear(512, 100)  # Number of relationship types
        )

        # Spatial feature encoder
        self.spatial_encoder = nn.Linear(4, 128)  # [x1, y1, x2, y2]

    def forward(self, visual_features, boxes):
        """
        Generate scene graph from visual features and bounding boxes

        Args:
            visual_features: (batch, num_objects, feature_dim)
            boxes: (batch, num_objects, 4) [x1, y1, x2, y2]
        """
        batch_size, num_objects, _ = visual_features.shape

        # Classify objects
        object_logits = self.object_classifier(visual_features)  # (batch, num_objects, vocab_size)
        object_classes = torch.argmax(object_logits, dim=-1)  # (batch, num_objects)

        # Encode spatial features
        spatial_features = self.spatial_encoder(boxes)  # (batch, num_objects, 128)

        # Compute relationships between all pairs
        relationships = []
        for i in range(num_objects):
            for j in range(num_objects):
                if i != j:
                    # Combine features of object i and j with spatial features
                    spatial_rel = torch.cat([
                        boxes[:, i, :],  # Box of subject
                        boxes[:, j, :]   # Box of object
                    ], dim=-1)  # (batch, 8)

                    spatial_rel_enc = self.spatial_encoder(spatial_rel)  # (batch, 128)

                    rel_features = torch.cat([
                        visual_features[:, i, :],  # Subject features
                        visual_features[:, j, :],  # Object features
                        spatial_rel_enc            # Spatial relationship
                    ], dim=-1)  # (batch, feature_dim * 2 + 128)

                    rel_logits = self.relation_classifier(rel_features)  # (batch, num_relations)
                    rel_class = torch.argmax(rel_logits, dim=-1)  # (batch,)

                    # Store relationship: (subject_idx, object_idx, relation_class)
                    relationships.append(torch.stack([
                        torch.full((batch_size,), i),
                        torch.full((batch_size,), j),
                        rel_class
                    ], dim=-1))

        if relationships:
            all_relationships = torch.stack(relationships, dim=1)  # (batch, num_pairs, 3)
        else:
            all_relationships = torch.zeros(batch_size, 0, 3, dtype=torch.long)

        return {
            'object_classes': object_classes,
            'relationships': all_relationships,
            'object_logits': object_logits
        }
```

### Language-Grounded Scene Understanding

```python
# grounded_scene_understanding.py
import torch
import torch.nn as nn

class GroundedSceneUnderstanding(nn.Module):
    def __init__(self, vision_model_name="openai/clip-vit-base-patch32",
                 language_model_name="meta-llama/Llama-2-7b-hf"):
        super(GroundedSceneUnderstanding, self).__init__()

        # Visual and language encoders
        from transformers import CLIPModel, LlamaModel
        self.clip_model = CLIPModel.from_pretrained(vision_model_name)
        self.llm = LlamaModel.from_pretrained(language_model_name)

        # Fusion module
        self.fusion = nn.MultiheadAttention(
            embed_dim=512,  # CLIP's feature dimension
            num_heads=8,
            batch_first=True
        )

        # Scene understanding head
        self.scene_classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 100)  # 100 scene categories
        )

        # Object detection head (simplified)
        self.object_detector = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 4 + 1)  # 4 for bbox + 1 for confidence
        )

    def forward(self, pixel_values, text_queries):
        """
        Understand scene based on visual input and language queries

        Args:
            pixel_values: Image tensor
            text_queries: Text queries about the scene
        """
        # Extract visual features
        visual_features = self.clip_model.get_image_features(pixel_values=pixel_values)

        # Process text queries
        # This is simplified - in practice, you would tokenize and encode the text
        # For now, we'll use the last hidden state of the language model as text features
        batch_size = pixel_values.size(0)

        # In a real implementation, you would process text_queries through the LLM
        # For this example, we'll create dummy text features
        text_features = torch.randn(batch_size, 1, 512, device=pixel_values.device)

        # Apply cross-modal attention
        attended_features, attention_weights = self.fusion(
            visual_features.unsqueeze(1),  # query: (batch, 1, 512)
            text_features,                 # key: (batch, seq_len, 512)
            text_features                  # value: (batch, seq_len, 512)
        )

        # Scene classification
        scene_logits = self.scene_classifier(attended_features.squeeze(1))  # (batch, 100)

        # Object detection (simplified)
        object_predictions = self.object_detector(visual_features)  # (batch, 4+1)

        return {
            'scene_classification': scene_logits,
            'object_predictions': object_predictions,
            'attention_weights': attention_weights,
            'fused_features': attended_features
        }

class MultimodalSceneGraph(nn.Module):
    def __init__(self, vocab_size=10000, feature_dim=512):
        super(MultimodalSceneGraph, self).__init__()

        self.feature_dim = feature_dim

        # Visual feature encoder
        self.visual_encoder = nn.Sequential(
            nn.Linear(768, feature_dim),  # CLIP visual features
            nn.LayerNorm(feature_dim),
            nn.ReLU()
        )

        # Language feature encoder
        self.language_encoder = nn.Sequential(
            nn.Linear(4096, feature_dim),  # LLM hidden size
            nn.LayerNorm(feature_dim),
            nn.ReLU()
        )

        # Multimodal fusion
        self.fusion = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )

        # Scene graph generation
        self.graph_generator = SceneGraphGenerator(vocab_size, feature_dim)

    def forward(self, visual_features, language_features, boxes):
        """
        Generate multimodal scene graph

        Args:
            visual_features: Visual features from CLIP
            language_features: Language features from LLM
            boxes: Bounding boxes for objects
        """
        batch_size = visual_features.size(0)

        # Encode features
        vis_encoded = self.visual_encoder(visual_features)  # (batch, num_objects, feature_dim)
        lang_encoded = self.language_encoder(language_features)  # (batch, seq_len, feature_dim)

        # Apply multimodal fusion
        fused_features, attention_weights = self.fusion(
            vis_encoded,  # query
            lang_encoded,  # key
            lang_encoded   # value
        )

        # Generate scene graph using fused features
        scene_graph = self.graph_generator(fused_features, boxes)

        return {
            **scene_graph,
            'attention_weights': attention_weights,
            'fused_features': fused_features
        }
```

## Implementation and Integration

### Complete Multimodal Perception System

```python
# complete_multimodal_system.py
import torch
import torch.nn as nn

class CompleteMultimodalPerception(nn.Module):
    def __init__(self, language_model_name="meta-llama/Llama-2-7b-hf"):
        super(CompleteMultimodalPerception, self).__init__()

        # Visual processing components
        self.visual_processor = VisualProcessor()

        # Language processing components
        self.language_processor = LanguageProcessor(language_model_name)

        # Multimodal fusion
        self.fusion = MultimodalFusion()

        # Spatial reasoning
        self.spatial_reasoner = SpatialReasoningModule()

        # Scene understanding
        self.scene_graph = MultimodalSceneGraph()

        # Grounding components
        self.grounding_system = AdvancedVisualGrounding()

    def forward(self, images, text_queries, boxes=None):
        """
        Complete multimodal perception pipeline

        Args:
            images: Batch of images
            text_queries: Text queries for grounding
            boxes: Optional bounding boxes for objects
        """
        # Visual processing
        visual_outputs = self.visual_processor(images)

        # Language processing
        tokenized = self.language_processor.tokenize_text(text_queries)
        language_outputs = self.language_processor(
            tokenized['input_ids'],
            tokenized['attention_mask'],
            visual_outputs['features']
        )

        # Multimodal fusion
        fusion_outputs = self.fusion(
            visual_outputs['features'],
            language_outputs['features']
        )

        # Spatial reasoning (if boxes provided)
        spatial_outputs = None
        if boxes is not None:
            spatial_outputs = self.spatial_reasoner(
                visual_outputs['spatial_features'],
                boxes
            )

        # Scene graph generation
        scene_outputs = self.scene_graph(
            visual_outputs['features'][:, 1:, :],  # Exclude global feature
            language_outputs['features'],
            boxes if boxes is not None else torch.zeros(1, 1, 4)  # Placeholder
        )

        # Visual grounding
        grounding_outputs = self.grounding_system(images, text_queries)

        return {
            'visual_features': visual_outputs,
            'language_features': language_outputs,
            'fused_features': fusion_outputs,
            'spatial_reasoning': spatial_outputs,
            'scene_graph': scene_outputs,
            'grounding': grounding_outputs
        }

    def predict_object_grounding(self, images, text_query):
        """
        Convenience method for object grounding
        """
        outputs = self(images, [text_query])

        # Extract grounding results
        grounding_scores = outputs['grounding']

        # Return the object in the image that best matches the text query
        best_match_idx = torch.argmax(grounding_scores[0])

        return {
            'best_match_score': grounding_scores[0][best_match_idx].item(),
            'all_scores': grounding_scores[0].tolist()
        }

# Example usage and training
class MultimodalPerceptionTrainer:
    def __init__(self, model, learning_rate=1e-4):
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()

    def train_step(self, images, text_queries, ground_truth_labels):
        """
        Single training step
        """
        self.model.train()

        # Forward pass
        outputs = self.model(images, text_queries)

        # Compute loss (this is simplified - in practice, you'd have specific losses
        # for each component: grounding loss, scene understanding loss, etc.)
        loss = torch.tensor(0.0, requires_grad=True)  # Placeholder

        # For demonstration, let's compute a simple grounding loss
        if 'grounding' in outputs:
            # This would compare with ground truth object associations
            pass

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item()
```

## Evaluation and Benchmarking

### Multimodal Perception Metrics

```python
# evaluation_metrics.py
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

class MultimodalEvaluation:
    def __init__(self):
        self.metrics = {}

    def evaluate_grounding_accuracy(self, predicted_boxes, ground_truth_boxes, iou_threshold=0.5):
        """Evaluate visual grounding accuracy"""
        ious = []
        correct_groundings = 0
        total_groundings = len(predicted_boxes)

        for pred_box, gt_box in zip(predicted_boxes, ground_truth_boxes):
            # Calculate IoU
            iou = self.calculate_iou(pred_box, gt_box)
            ious.append(iou)

            if iou >= iou_threshold:
                correct_groundings += 1

        accuracy = correct_groundings / total_groundings if total_groundings > 0 else 0
        mean_iou = np.mean(ious) if ious else 0

        return {
            'grounding_accuracy': accuracy,
            'mean_iou': mean_iou,
            'ious': ious
        }

    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union"""
        # box format: [x1, y1, x2, y2]
        x1_inter = max(box1[0], box2[0])
        y1_inter = max(box1[1], box2[1])
        x2_inter = min(box1[2], box2[2])
        y2_inter = min(box1[3], box2[3])

        if x2_inter <= x1_inter or y2_inter <= y1_inter:
            return 0.0

        inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = area1 + area2 - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

    def evaluate_language_understanding(self, predicted_actions, ground_truth_actions, language_commands):
        """Evaluate how well the system understands language in context of perception"""
        # Calculate action accuracy for each language command
        action_accuracy = accuracy_score(
            ground_truth_actions.argmax(axis=1),
            predicted_actions.argmax(axis=1)
        )

        # Calculate command-specific accuracy
        command_accuracy = {}
        for cmd in set(language_commands):
            cmd_mask = np.array(language_commands) == cmd
            if np.sum(cmd_mask) > 0:
                cmd_acc = accuracy_score(
                    np.array(ground_truth_actions)[cmd_mask].argmax(axis=1),
                    np.array(predicted_actions)[cmd_mask].argmax(axis=1)
                )
                command_accuracy[cmd] = cmd_acc

        return {
            'overall_accuracy': action_accuracy,
            'command_accuracy': command_accuracy
        }

    def generate_comprehensive_report(self, model, test_dataset):
        """Generate comprehensive evaluation report"""
        all_grounding_results = []
        all_language_results = []

        for batch in test_dataset:
            # Run inference
            with torch.no_grad():
                outputs = model(
                    batch['images'],
                    batch['text_queries'],
                    batch.get('boxes')
                )

            # Evaluate grounding if ground truth is available
            if 'ground_truth_boxes' in batch:
                grounding_eval = self.evaluate_grounding_accuracy(
                    outputs['grounding_predictions'],
                    batch['ground_truth_boxes']
                )
                all_grounding_results.append(grounding_eval)

            # Evaluate language understanding
            if 'ground_truth_actions' in batch:
                language_eval = self.evaluate_language_understanding(
                    outputs['predicted_actions'],
                    batch['ground_truth_actions'],
                    batch['language_commands']
                )
                all_language_results.append(language_eval)

        # Aggregate results
        avg_grounding_acc = np.mean([r['grounding_accuracy'] for r in all_grounding_results])
        avg_mean_iou = np.mean([r['mean_iou'] for r in all_grounding_results])

        report = f"""
        Multimodal Perception Evaluation Report
        ======================================

        Visual Grounding Performance:
        - Grounding Accuracy: {avg_grounding_acc:.3f}
        - Mean IoU: {avg_mean_iou:.3f}

        Language Understanding:
        - Overall Action Accuracy: {(sum(r['overall_accuracy'] for r in all_language_results) / len(all_language_results)):.3f}

        System Capabilities:
        """

        if avg_grounding_acc > 0.7:
            report += "- Strong visual grounding capabilities\n"
        elif avg_grounding_acc > 0.5:
            report += "- Moderate visual grounding capabilities\n"
        else:
            report += "- Visual grounding needs improvement\n"

        if avg_mean_iou > 0.5:
            report += "- Good spatial precision\n"
        else:
            report += "- Spatial precision needs improvement\n"

        return report
```

## Summary

Multimodal perception forms the foundation of Vision-Language-Action systems, enabling robots to understand their environment through both visual and linguistic modalities. The integration of visual processing, language understanding, and spatial reasoning creates a comprehensive system that can ground language commands to specific visual objects and locations.

Key components include:
- Visual processing pipelines that extract meaningful features from images
- Language processing systems that understand natural language commands
- Multimodal fusion mechanisms that combine information from different modalities
- Spatial reasoning modules that understand object relationships and positions
- Grounding systems that connect language to visual elements

These components work together to enable robots to perceive and understand their environment in a human-like manner, setting the stage for language-guided action and manipulation systems in the next section.