---
sidebar_position: 4
---

# Language-Guided Manipulation: From Commands to Actions

## Learning Objectives

By the end of this section, you will be able to:
- Understand the principles of language-guided robotic manipulation
- Implement natural language processing for robotic command interpretation
- Create action planning systems that execute language commands
- Design skill-based manipulation frameworks with language grounding
- Integrate perception systems with language-guided control
- Evaluate language-guided manipulation performance and robustness

## Introduction to Language-Guided Manipulation

**Language-guided manipulation** enables robots to understand and execute natural language commands for object manipulation tasks. This capability allows humans to interact with robots using intuitive, high-level instructions rather than low-level programming or teleoperation.

Language-guided manipulation involves:
- **Natural Language Understanding**: Interpreting human commands
- **Action Planning**: Converting language to executable actions
- **Perception Integration**: Connecting language to visual objects
- **Skill Execution**: Performing manipulation tasks based on commands
- **Feedback and Correction**: Handling ambiguous or incorrect commands

## Language Command Processing

### Command Parsing and Understanding

```python
# command_parser.py
import torch
import torch.nn as nn
import re
from transformers import LlamaModel, LlamaTokenizer
from typing import Dict, List, Tuple, Optional

class CommandParser(nn.Module):
    def __init__(self, model_name="meta-llama/Llama-2-7b-hf"):
        super(CommandParser, self).__init__()

        self.tokenizer = LlamaTokenizer.from_pretrained(model_name)
        self.llm = LlamaModel.from_pretrained(model_name)

        # Add special tokens for manipulation commands
        special_tokens = {
            'additional_special_tokens': [
                '<ACTION>', '</ACTION>',
                '<OBJECT>', '</OBJECT>',
                '<LOCATION>', '</LOCATION>',
                '<GRASP>', '</GRASP>',
                '<PLACE>', '</PLACE>',
                '<MOVE>', '</MOVE>'
            ]
        }
        self.tokenizer.add_special_tokens(special_tokens)

        # Command classification head
        self.command_classifier = nn.Sequential(
            nn.Linear(self.llm.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 10)  # 10 common manipulation actions
        )

        # Argument extraction head
        self.argument_extractor = nn.Sequential(
            nn.Linear(self.llm.config.hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, 3)  # object, location, grasp type
        )

        # Action sequence generator
        self.action_generator = nn.Sequential(
            nn.Linear(self.llm.config.hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 50)  # Maximum 50 actions in sequence
        )

    def parse_command(self, command: str) -> Dict:
        """
        Parse a natural language command into structured components
        """
        # Tokenize the command
        inputs = self.tokenizer(
            command,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=128
        )

        # Get language features
        outputs = self.llm(**inputs)
        hidden_states = outputs.last_hidden_state  # (1, seq_len, hidden_size)

        # Get command classification
        command_logits = self.command_classifier(hidden_states[:, -1, :])  # Use last token
        command_type = torch.argmax(command_logits, dim=-1).item()

        # Extract arguments
        arguments = self.argument_extractor(hidden_states[:, -1, :])

        # Generate action sequence
        action_seq = self.action_generator(hidden_states[:, -1, :])

        return {
            'command_type': command_type,
            'arguments': arguments,
            'action_sequence': action_seq,
            'raw_command': command
        }

    def extract_entities(self, command: str) -> Dict[str, List[str]]:
        """
        Extract objects, locations, and other entities from command
        """
        # Simple pattern matching for demonstration
        # In practice, use more sophisticated NLP techniques
        entities = {
            'objects': [],
            'locations': [],
            'actions': [],
            'modifiers': []
        }

        # Common objects
        object_patterns = [
            r'\b(block|cube|ball|cup|bottle|box|container|toy|book|pen|phone|keys|apple|banana)\b',
            r'\b(red|blue|green|yellow|large|small|big|tiny|heavy|light)\b'
        ]

        # Common locations
        location_patterns = [
            r'\b(table|shelf|counter|floor|box|bin|cabinet|drawer|fridge|microwave)\b',
            r'\b(left|right|center|middle|front|back|near|beside|on|in|under|above)\b'
        ]

        # Common actions
        action_patterns = [
            r'\b(pick|grasp|take|lift|hold|move|place|put|set|drop|release|transfer)\b'
        ]

        for pattern in object_patterns[0].split('|'):
            pattern = pattern.strip('()')
            matches = re.findall(pattern, command, re.IGNORECASE)
            entities['objects'].extend(matches)

        for pattern in location_patterns[0].split('|'):
            pattern = pattern.strip('()')
            matches = re.findall(pattern, command, re.IGNORECASE)
            entities['locations'].extend(matches)

        for pattern in action_patterns[0].split('|'):
            pattern = pattern.strip('()')
            matches = re.findall(pattern, command, re.IGNORECASE)
            entities['actions'].extend(matches)

        return entities

class AdvancedCommandParser(nn.Module):
    def __init__(self, vocab_size=32000, hidden_dim=4096):
        super(AdvancedCommandParser, self).__init__()

        self.hidden_dim = hidden_dim

        # Semantic role labeling for command understanding
        self.semantic_analyzer = nn.Sequential(
            nn.Linear(hidden_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 20)  # 20 semantic roles (agent, theme, goal, etc.)
        )

        # Task decomposition network
        self.task_decomposer = nn.Sequential(
            nn.Linear(hidden_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 100)  # Maximum 100 subtasks
        )

        # Grasp type classifier
        self.grasp_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 5)  # 5 grasp types: power, precision, pinch, etc.
        )

        # Motion primitive selector
        self.motion_selector = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 20)  # 20 common motion primitives
        )

    def forward(self, command_features, visual_context=None):
        """
        Analyze command and generate structured representation

        Args:
            command_features: Language features from LLM
            visual_context: Optional visual context features
        """
        batch_size = command_features.size(0)

        # Semantic role analysis
        semantic_roles = self.semantic_analyzer(command_features)  # (batch, 20)

        # Task decomposition
        task_decomposition = self.task_decomposer(command_features)  # (batch, 100)
        # Apply softmax and get top tasks
        task_probs = torch.softmax(task_decomposition, dim=-1)
        top_tasks = torch.topk(task_probs, k=5, dim=-1)  # Top 5 subtasks

        # Grasp type prediction
        grasp_types = self.grasp_classifier(command_features)  # (batch, 5)

        # Motion primitive selection
        motion_primitives = self.motion_selector(command_features)  # (batch, 20)

        return {
            'semantic_roles': semantic_roles,
            'task_decomposition': {
                'tasks': top_tasks.indices,
                'probabilities': top_tasks.values
            },
            'grasp_types': grasp_types,
            'motion_primitives': motion_primitives
        }
```

### Command-to-Action Mapping

```python
# command_action_mapping.py
import torch
import torch.nn as nn
from enum import Enum

class ManipulationAction(Enum):
    PICK = 0
    PLACE = 1
    MOVE = 2
    GRASP = 3
    RELEASE = 4
    APPROACH = 5
    RETRACT = 6
    ALIGN = 7
    INSERT = 8
    EXTRACT = 9

class CommandActionMapper(nn.Module):
    def __init__(self, hidden_dim=4096, action_dim=7):
        super(CommandActionMapper, self).__init__()

        self.action_dim = action_dim

        # Command embedding to action mapping
        self.command_to_action = nn.Sequential(
            nn.Linear(hidden_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim * 10)  # 10 time steps
        )

        # Action type classifier
        self.action_type_classifier = nn.Linear(hidden_dim, len(ManipulationAction))

        # Object-specific action modifier
        self.object_action_modifier = nn.Sequential(
            nn.Linear(hidden_dim + 512, 512),  # command + object features
            nn.ReLU(),
            nn.Linear(512, action_dim)
        )

    def forward(self, command_features, object_features=None):
        """
        Map command features to robot actions

        Args:
            command_features: Language features from command
            object_features: Optional object-specific features
        """
        batch_size = command_features.size(0)

        # Map command to sequence of actions
        action_sequence = self.command_to_action(command_features)
        action_sequence = action_sequence.view(batch_size, 10, self.action_dim)  # (batch, 10, action_dim)

        # Classify action type
        action_types = self.action_type_classifier(command_features)  # (batch, num_action_types)

        # If object features provided, modify actions accordingly
        if object_features is not None:
            combined_features = torch.cat([command_features, object_features], dim=-1)
            action_modifiers = self.object_action_modifier(combined_features)  # (batch, action_dim)

            # Apply modifiers to the action sequence
            modified_actions = action_sequence + action_modifiers.unsqueeze(1)  # Broadcast to sequence
        else:
            modified_actions = action_sequence

        return {
            'action_sequence': modified_actions,
            'action_types': action_types,
            'predicted_actions': modified_actions[:, 0, :]  # First action in sequence
        }

class HierarchicalCommandProcessor(nn.Module):
    def __init__(self, hidden_dim=4096):
        super(HierarchicalCommandProcessor, self).__init__()

        # High-level task planner
        self.task_planner = nn.Sequential(
            nn.Linear(hidden_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 50)  # 50 possible high-level tasks
        )

        # Skill selector
        self.skill_selector = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 20)  # 20 manipulation skills
        )

        # Skill parameter generator
        self.skill_parameter_generator = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 100)  # 100 possible skill parameters
        )

        # Skill execution sequence
        self.skill_execution_planner = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 50),  # Sequence of 50 skill steps
            nn.Softmax(dim=-1)
        )

    def forward(self, command_features):
        """
        Process command hierarchically: task -> skills -> parameters -> execution
        """
        batch_size = command_features.size(0)

        # Plan high-level task
        task_plan = self.task_planner(command_features)  # (batch, 50)
        selected_task = torch.argmax(task_plan, dim=-1)  # (batch,)

        # Select appropriate skills
        skill_scores = self.skill_selector(command_features)  # (batch, 20)
        selected_skills = torch.topk(skill_scores, k=3, dim=-1)  # Top 3 skills

        # Generate skill parameters
        skill_params = self.skill_parameter_generator(command_features)  # (batch, 100)

        # Plan skill execution sequence
        execution_sequence = self.skill_execution_planner(command_features)  # (batch, 50)

        return {
            'selected_task': selected_task,
            'selected_skills': selected_skills.indices,
            'skill_parameters': skill_params,
            'execution_sequence': execution_sequence
        }
```

## Action Planning and Execution

### Skill-Based Manipulation Framework

```python
# skill_framework.py
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple

class ManipulationSkill:
    def __init__(self, name: str, skill_id: int):
        self.name = name
        self.skill_id = skill_id
        self.primitive_actions = []
        self.preconditions = []
        self.postconditions = []

    def execute(self, robot_state, target_object, target_location):
        """Execute the skill with given parameters"""
        raise NotImplementedError

class PickSkill(ManipulationSkill):
    def __init__(self):
        super().__init__("pick", 0)
        self.preconditions = ["object_visible", "gripper_open", "not_holding_object"]
        self.postconditions = ["object_grasped", "gripper_closed"]

    def execute(self, robot_state, target_object, target_location=None):
        """Execute pick action"""
        # Implementation would interface with robot controller
        actions = [
            {"type": "approach", "object": target_object, "offset": 0.1},
            {"type": "grasp", "object": target_object},
            {"type": "lift", "height": 0.1}
        ]
        return actions

class PlaceSkill(ManipulationSkill):
    def __init__(self):
        super().__init__("place", 1)
        self.preconditions = ["holding_object", "gripper_closed"]
        self.postconditions = ["object_released", "gripper_open", "object_placed"]

    def execute(self, robot_state, target_object, target_location):
        """Execute place action"""
        actions = [
            {"type": "approach", "location": target_location, "offset": 0.1},
            {"type": "align", "location": target_location},
            {"type": "release", "location": target_location},
            {"type": "retract", "distance": 0.1}
        ]
        return actions

class SkillLibrary(nn.Module):
    def __init__(self, skill_dim=256):
        super(SkillLibrary, self).__init__()

        self.skills = {
            0: PickSkill(),
            1: PlaceSkill(),
            # Add more skills as needed
        }

        # Skill embedding network
        self.skill_embedder = nn.Sequential(
            nn.Linear(4096, 512),  # Language features -> skill features
            nn.ReLU(),
            nn.Linear(512, skill_dim),
            nn.LayerNorm(skill_dim)
        )

        # Skill selector
        self.skill_selector = nn.Sequential(
            nn.Linear(skill_dim, 256),
            nn.ReLU(),
            nn.Linear(256, len(self.skills))
        )

        # Skill parameter generator
        self.skill_parameter_generator = nn.Sequential(
            nn.Linear(skill_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 50)  # 50 possible parameters
        )

    def select_skill(self, command_features):
        """Select appropriate skill based on command"""
        skill_features = self.skill_embedder(command_features)
        skill_logits = self.skill_selector(skill_features)
        skill_probs = torch.softmax(skill_logits, dim=-1)

        # Get most likely skill
        selected_skill_id = torch.argmax(skill_probs, dim=-1)

        return {
            'skill_id': selected_skill_id.item(),
            'skill_probabilities': skill_probs,
            'skill_features': skill_features
        }

    def generate_skill_parameters(self, command_features, skill_features):
        """Generate parameters for selected skill"""
        combined_features = torch.cat([command_features, skill_features], dim=-1)
        parameters = self.skill_parameter_generator(combined_features)

        return parameters

class SkillExecutor(nn.Module):
    def __init__(self):
        super(SkillExecutor, self).__init__()

        self.skill_library = SkillLibrary()

    def forward(self, command_features, visual_objects, target_object_id, target_location=None):
        """
        Execute manipulation skill based on command and visual context

        Args:
            command_features: Language features from command
            visual_objects: Detected objects with features
            target_object_id: ID of target object to manipulate
            target_location: Optional target location for placement
        """
        # Select skill based on command
        skill_selection = self.skill_library.select_skill(command_features)
        skill_id = skill_selection['skill_id']

        # Get target object features
        if target_object_id < len(visual_objects):
            target_features = visual_objects[target_object_id]['features']
        else:
            target_features = torch.zeros(512)  # Default features

        # Generate skill parameters
        skill_params = self.skill_library.generate_skill_parameters(
            command_features,
            skill_selection['skill_features']
        )

        # Execute the selected skill
        selected_skill = self.skill_library.skills[skill_id]

        # In a real implementation, this would interface with robot controller
        # For this example, we'll return a structured action plan
        action_plan = {
            'skill_name': selected_skill.name,
            'skill_id': skill_id,
            'target_object': target_object_id,
            'target_location': target_location,
            'parameters': skill_params,
            'preconditions': selected_skill.preconditions,
            'postconditions': selected_skill.postconditions
        }

        return action_plan
```

### Motion Planning Integration

```python
# motion_planning_integration.py
import torch
import torch.nn as nn
import numpy as np

class MotionPlanner(nn.Module):
    def __init__(self, action_dim=7):
        super(MotionPlanner, self).__init__()

        self.action_dim = action_dim

        # Trajectory generator
        self.trajectory_generator = nn.Sequential(
            nn.Linear(512, 512),  # Skill + object features
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 100 * action_dim),  # 100 waypoints
            nn.Tanh()
        )

        # Collision avoidance network
        self.collision_avoider = nn.Sequential(
            nn.Linear(512 + 64, 256),  # skill + environment features
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )

        # Gripper controller
        self.gripper_controller = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),  # gripper width
            nn.Sigmoid()
        )

    def plan_trajectory(self, skill_features, object_features, environment_features):
        """
        Plan motion trajectory based on skill and environment

        Args:
            skill_features: Features of selected manipulation skill
            object_features: Features of target object
            environment_features: Features of environment/obstacles
        """
        # Combine features for trajectory planning
        combined_features = torch.cat([skill_features, object_features], dim=-1)

        # Generate trajectory
        trajectory_flat = self.trajectory_generator(combined_features)
        trajectory = trajectory_flat.view(-1, 100, self.action_dim)  # (batch, 100, action_dim)

        # Apply collision avoidance
        env_augmented = torch.cat([skill_features, environment_features], dim=-1)
        collision_adjustments = self.collision_avoider(env_augmented)

        # Adjust trajectory with collision avoidance
        adjusted_trajectory = trajectory + collision_adjustments.unsqueeze(1)

        # Generate gripper commands
        gripper_commands = self.gripper_controller(skill_features)  # (batch, 1)

        return {
            'trajectory': adjusted_trajectory,
            'gripper_commands': gripper_commands,
            'collision_adjustments': collision_adjustments
        }

class LanguageGuidedMotionPlanner(nn.Module):
    def __init__(self, hidden_dim=4096, action_dim=7):
        super(LanguageGuidedMotionPlanner, self).__init__()

        self.action_dim = action_dim

        # Language-to-motion mapping
        self.lang_to_motion = nn.Sequential(
            nn.Linear(hidden_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim * 10)  # 10 action steps
        )

        # Waypoint generator
        self.waypoint_generator = nn.Sequential(
            nn.Linear(512, 256),  # Visual + language features
            nn.ReLU(),
            nn.Linear(256, 50 * 3),  # 50 waypoints * 3D coordinates
            nn.Tanh()
        )

        # Motion primitive selector
        self.motion_primitive_selector = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 15)  # 15 motion primitives
        )

    def forward(self, command_features, visual_features, target_position):
        """
        Generate motion plan from language command and visual context

        Args:
            command_features: Language features from command
            visual_features: Visual features of environment
            target_position: Target position for manipulation
        """
        batch_size = command_features.size(0)

        # Generate motion sequence from language
        motion_sequence = self.lang_to_motion(command_features)
        motion_sequence = motion_sequence.view(batch_size, 10, self.action_dim)

        # Generate waypoints
        combined_features = torch.cat([command_features, visual_features], dim=-1)
        waypoints_flat = self.waypoint_generator(combined_features)
        waypoints = waypoints_flat.view(batch_size, 50, 3)  # (batch, 50, 3)

        # Select motion primitives
        motion_primitives = self.motion_primitive_selector(command_features)

        # Adjust trajectory to target
        target_expanded = target_position.unsqueeze(1).expand(-1, 50, -1)  # (batch, 50, 3)
        final_waypoints = waypoints * 0.5 + target_expanded * 0.5  # Blend with target

        return {
            'motion_sequence': motion_sequence,
            'waypoints': final_waypoints,
            'motion_primitives': motion_primitives,
            'target_position': target_position
        }
```

## Perception-Action Integration

### Object Grounding and Manipulation

```python
# perception_action_integration.py
import torch
import torch.nn as nn

class ObjectGroundingAndManipulation(nn.Module):
    def __init__(self, hidden_dim=4096, action_dim=7):
        super(ObjectGroundingAndManipulation, self).__init__()

        self.action_dim = action_dim

        # Object detection and classification
        self.object_detector = nn.Sequential(
            nn.Linear(512, 256),  # Visual features
            nn.ReLU(),
            nn.Linear(256, 100)   # 100 object classes
        )

        # Object-language grounding
        self.grounding_network = nn.Sequential(
            nn.Linear(512 + 4096, 512),  # visual + language features
            nn.ReLU(),
            nn.Linear(512, 1),  # Grounding score
            nn.Sigmoid()
        )

        # Grasp pose prediction
        self.grasp_predictor = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 7),  # 7-DOF grasp pose (position + quaternion)
        )

        # Manipulation action generator
        self.action_generator = nn.Sequential(
            nn.Linear(512 + 4096 + 7, 512),  # visual + language + grasp
            nn.ReLU(),
            nn.Linear(512, action_dim * 15),  # 15 action steps
            nn.Tanh()
        )

    def forward(self, visual_features, language_features, object_boxes):
        """
        Integrate perception and action for language-guided manipulation

        Args:
            visual_features: Visual features from scene
            language_features: Language features from command
            object_boxes: Bounding boxes of detected objects
        """
        batch_size, num_objects, vis_dim = visual_features.shape

        # Detect objects
        object_logits = self.object_detector(visual_features)  # (batch, num_objects, 100)
        object_classes = torch.argmax(object_logits, dim=-1)  # (batch, num_objects)

        # Ground language to objects
        grounded_scores = []
        for i in range(num_objects):
            # Combine visual and language features for grounding
            vis_lang_features = torch.cat([
                visual_features[:, i, :],
                language_features.expand(-1, vis_dim).contiguous()
            ], dim=-1)
            score = self.grounding_network(vis_lang_features)  # (batch, 1)
            grounded_scores.append(score)

        grounding_scores = torch.cat(grounded_scores, dim=1)  # (batch, num_objects)
        most_likely_object = torch.argmax(grounding_scores, dim=1)  # (batch,)

        # Predict grasp pose for the selected object
        selected_object_features = visual_features[torch.arange(batch_size), most_likely_object]
        grasp_pose = self.grasp_predictor(selected_object_features)  # (batch, 7)

        # Generate manipulation actions
        combined_features = torch.cat([
            selected_object_features,
            language_features,
            grasp_pose
        ], dim=1)
        action_sequence = self.action_generator(combined_features)
        action_sequence = action_sequence.view(batch_size, 15, self.action_dim)

        return {
            'object_classes': object_classes,
            'grounding_scores': grounding_scores,
            'selected_object': most_likely_object,
            'grasp_pose': grasp_pose,
            'action_sequence': action_sequence,
            'object_boxes': object_boxes
        }

class MultimodalManipulationController(nn.Module):
    def __init__(self, hidden_dim=4096, action_dim=7):
        super(MultimodalManipulationController, self).__init__()

        # Visual processing
        self.visual_processor = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256)
        )

        # Language processing
        self.language_processor = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256)
        )

        # Multimodal fusion
        self.fusion = nn.MultiheadAttention(
            embed_dim=256,
            num_heads=8,
            batch_first=True
        )

        # Manipulation policy
        self.manipulation_policy = nn.Sequential(
            nn.Linear(256 * 2, 512),  # fused features * 2 (current + goal)
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )

        # Skill sequence planner
        self.skill_planner = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 50)  # Maximum 50 skills in sequence
        )

    def forward(self, visual_features, language_features, current_state, goal_state):
        """
        Generate manipulation actions from multimodal inputs

        Args:
            visual_features: Current visual scene features
            language_features: Language command features
            current_state: Current robot state
            goal_state: Target state
        """
        batch_size = visual_features.size(0)

        # Process visual and language features
        vis_processed = self.visual_processor(visual_features)
        lang_processed = self.language_processor(language_features)

        # Multimodal fusion
        fused_features, attention_weights = self.fusion(
            vis_processed.unsqueeze(1),  # query
            lang_processed.unsqueeze(1), # key
            lang_processed.unsqueeze(1)  # value
        )
        fused_features = fused_features.squeeze(1)  # (batch, 256)

        # Combine with state information
        state_features = torch.cat([current_state, goal_state], dim=-1)
        policy_input = torch.cat([fused_features, state_features], dim=-1)

        # Generate action
        action = self.manipulation_policy(policy_input)

        # Plan skill sequence
        skill_sequence = self.skill_planner(fused_features)

        return {
            'action': action,
            'skill_sequence': skill_sequence,
            'fused_features': fused_features,
            'attention_weights': attention_weights
        }
```

## Implementation Examples

### Complete Language-Guided Manipulation System

```python
# complete_manipulation_system.py
import torch
import torch.nn as nn

class CompleteLanguageGuidedManipulation(nn.Module):
    def __init__(self, language_model_name="meta-llama/Llama-2-7b-hf"):
        super(CompleteLanguageGuidedManipulation, self).__init__()

        # Command processing
        self.command_parser = AdvancedCommandParser()

        # Skill framework
        self.skill_executor = SkillExecutor()

        # Motion planning
        self.motion_planner = LanguageGuidedMotionPlanner()

        # Perception-action integration
        self.perception_action = ObjectGroundingAndManipulation()

        # Multimodal controller
        self.controller = MultimodalManipulationController()

    def forward(self,
                images,
                command_text,
                current_robot_state,
                detected_objects=None,
                target_location=None):
        """
        Complete language-guided manipulation pipeline

        Args:
            images: Current scene images
            command_text: Natural language command
            current_robot_state: Current robot joint positions/velocities
            detected_objects: Optional pre-detected objects
            target_location: Optional target location
        """
        # This is a high-level pipeline - in practice, you would have:
        # 1. Process the language command
        # 2. Analyze the visual scene
        # 3. Ground language to visual objects
        # 4. Plan manipulation actions
        # 5. Execute the plan

        # For this example, we'll return a structured manipulation plan
        manipulation_plan = {
            'command': command_text,
            'parsed_command': self.parse_command(command_text),
            'detected_objects': detected_objects,
            'target_object': self.select_target_object(command_text, detected_objects),
            'manipulation_action': self.determine_manipulation_action(command_text),
            'motion_plan': self.generate_motion_plan(command_text, detected_objects),
            'execution_status': 'planned'
        }

        return manipulation_plan

    def parse_command(self, command_text):
        """Parse command text (simplified for this example)"""
        # In practice, this would use the command_parser
        return {
            'raw_command': command_text,
            'command_type': 'pick_and_place',  # Simplified
            'target_object': self.extract_object(command_text),
            'target_location': self.extract_location(command_text)
        }

    def extract_object(self, command_text):
        """Extract target object from command"""
        # Simple keyword matching for demonstration
        objects = ['block', 'cup', 'bottle', 'box', 'ball']
        for obj in objects:
            if obj in command_text.lower():
                return obj
        return 'object'

    def extract_location(self, command_text):
        """Extract target location from command"""
        locations = ['table', 'shelf', 'box', 'bin']
        for loc in locations:
            if loc in command_text.lower():
                return loc
        return 'location'

    def select_target_object(self, command_text, detected_objects):
        """Select target object based on command and detection"""
        if detected_objects:
            # In practice, use grounding network
            return detected_objects[0]  # First detected object
        return None

    def determine_manipulation_action(self, command_text):
        """Determine manipulation action from command"""
        if 'pick' in command_text.lower() or 'grasp' in command_text.lower():
            return 'pick'
        elif 'place' in command_text.lower() or 'put' in command_text.lower():
            return 'place'
        elif 'move' in command_text.lower():
            return 'move'
        return 'manipulate'

    def generate_motion_plan(self, command_text, detected_objects):
        """Generate motion plan for manipulation"""
        return {
            'waypoints': [],  # Planned waypoints
            'gripper_commands': [],  # Gripper actions
            'collision_free': True  # Collision check result
        }

# Example usage
def example_usage():
    """Example of how to use the language-guided manipulation system"""
    # Initialize the system
    manipulation_system = CompleteLanguageGuidedManipulation()

    # Example command
    command = "Pick up the red block and place it on the table"

    # Example robot state (simplified)
    robot_state = torch.randn(1, 7)  # 7-DOF robot arm

    # Example detected objects (simplified)
    detected_objects = [
        {'class': 'block', 'position': [0.5, 0.3, 0.1], 'color': 'red'},
        {'class': 'cup', 'position': [0.6, 0.4, 0.1], 'color': 'blue'}
    ]

    # Generate manipulation plan
    plan = manipulation_system(
        images=None,  # Would be actual images in practice
        command_text=command,
        current_robot_state=robot_state,
        detected_objects=detected_objects
    )

    print(f"Command: {command}")
    print(f"Action: {plan['manipulation_action']}")
    print(f"Target: {plan['target_object']}")
    print(f"Status: {plan['execution_status']}")

if __name__ == "__main__":
    example_usage()
```

### Real-World Integration Example

```python
# real_world_integration.py
import rospy
from geometry_msgs.msg import Pose, Point
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np

class RealWorldManipulationInterface:
    def __init__(self):
        # ROS publishers/subscribers
        self.bridge = CvBridge()

        # Subscribe to camera feed
        self.image_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, self.image_callback)

        # Subscribe to robot state
        self.state_sub = rospy.Subscriber("/joint_states", JointState, self.state_callback)

        # Publisher for robot commands
        self.command_pub = rospy.Publisher("/joint_group_position_controller/command", JointTrajectory, queue_size=10)

        # Language-guided manipulation system
        self.manipulation_system = CompleteLanguageGuidedManipulation()

        # Current state storage
        self.current_image = None
        self.current_state = None
        self.object_detector = None  # Would be initialized with actual detector

    def image_callback(self, msg):
        """Callback for camera images"""
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            # Store for processing
            self.current_image = cv_image

            # Run object detection if needed
            if self.object_detector:
                detected_objects = self.object_detector.detect(cv_image)
                self.process_detected_objects(detected_objects)

        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")

    def state_callback(self, msg):
        """Callback for robot state"""
        # Process joint states
        self.current_state = np.array(msg.position)

    def execute_language_command(self, command_text):
        """Execute a language command in the real world"""
        if self.current_image is None or self.current_state is None:
            rospy.logwarn("Waiting for image and state data...")
            return False

        # Convert current state to tensor
        state_tensor = torch.FloatTensor(self.current_state).unsqueeze(0)

        # In a real implementation, you would:
        # 1. Process the current image through the perception pipeline
        # 2. Run the manipulation system to generate actions
        # 3. Execute the actions on the real robot

        # For this example, we'll return a placeholder
        rospy.loginfo(f"Processing command: {command_text}")

        # Generate manipulation plan
        plan = self.manipulation_system(
            images=None,  # Would process current_image
            command_text=command_text,
            current_robot_state=state_tensor,
            detected_objects=self.get_detected_objects()  # Would come from perception
        )

        # Execute the plan (simplified)
        self.execute_manipulation_plan(plan)

        return True

    def get_detected_objects(self):
        """Get currently detected objects (placeholder)"""
        # In practice, this would come from real object detection
        return [
            {'class': 'block', 'position': [0.5, 0.3, 0.1], 'features': torch.randn(512)},
            {'class': 'cup', 'position': [0.6, 0.4, 0.1], 'features': torch.randn(512)}
        ]

    def execute_manipulation_plan(self, plan):
        """Execute the generated manipulation plan"""
        rospy.loginfo(f"Executing manipulation plan: {plan['manipulation_action']}")

        # In practice, this would send actual commands to the robot
        # For now, we'll just log the action
        if plan['manipulation_action'] == 'pick':
            self.execute_pick_action(plan['target_object'])
        elif plan['manipulation_action'] == 'place':
            self.execute_place_action(plan['target_object'], plan.get('target_location'))
        else:
            rospy.logwarn(f"Unknown manipulation action: {plan['manipulation_action']}")

    def execute_pick_action(self, target_object):
        """Execute pick action"""
        rospy.loginfo(f"Picking up {target_object}")
        # Implementation would move robot to object and grasp it

    def execute_place_action(self, target_object, target_location):
        """Execute place action"""
        rospy.loginfo(f"Placing {target_object} at {target_location}")
        # Implementation would move robot to location and release object
```

## Evaluation and Performance Metrics

### Manipulation Performance Evaluation

```python
# manipulation_evaluation.py
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

class ManipulationEvaluator:
    def __init__(self):
        self.metrics = {}

    def evaluate_command_understanding(self, predicted_commands, ground_truth_commands):
        """Evaluate how well the system understands language commands"""
        command_accuracy = accuracy_score(
            [cmd['action_type'] for cmd in ground_truth_commands],
            [cmd['action_type'] for cmd in predicted_commands]
        )

        # Calculate precision, recall, F1 for each command type
        gt_actions = [cmd['action_type'] for cmd in ground_truth_commands]
        pred_actions = [cmd['action_type'] for cmd in predicted_commands]

        precision, recall, f1, support = precision_recall_fscore_support(
            gt_actions, pred_actions, average='weighted'
        )

        return {
            'command_accuracy': command_accuracy,
            'command_precision': precision,
            'command_recall': recall,
            'command_f1': f1
        }

    def evaluate_object_grounding(self, predicted_groundings, ground_truth_groundings, iou_threshold=0.5):
        """Evaluate how well the system grounds language to objects"""
        correct_groundings = 0
        total_groundings = len(predicted_groundings)

        for pred, gt in zip(predicted_groundings, ground_truth_groundings):
            if self.calculate_iou(pred['bbox'], gt['bbox']) >= iou_threshold:
                correct_groundings += 1

        grounding_accuracy = correct_groundings / total_groundings if total_groundings > 0 else 0

        return {
            'grounding_accuracy': grounding_accuracy,
            'iou_threshold': iou_threshold
        }

    def evaluate_manipulation_success(self, executed_actions, expected_outcomes):
        """Evaluate manipulation task success"""
        successful_tasks = 0
        total_tasks = len(executed_actions)

        for action, expected in zip(executed_actions, expected_outcomes):
            if self.check_task_success(action, expected):
                successful_tasks += 1

        success_rate = successful_tasks / total_tasks if total_tasks > 0 else 0

        return {
            'success_rate': success_rate,
            'successful_tasks': successful_tasks,
            'total_tasks': total_tasks
        }

    def check_task_success(self, action_result, expected_result):
        """Check if manipulation task was successful"""
        # This would compare actual vs expected outcomes
        # For example: object moved to correct location, grasp successful, etc.
        return action_result.get('success', False) and expected_result.get('success', False)

    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union"""
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

    def generate_performance_report(self, system, test_dataset):
        """Generate comprehensive performance report"""
        command_metrics = self.evaluate_command_understanding(
            system.predict_commands(test_dataset['commands']),
            test_dataset['ground_truth_commands']
        )

        grounding_metrics = self.evaluate_object_grounding(
            system.predict_groundings(test_dataset['images'], test_dataset['commands']),
            test_dataset['ground_truth_groundings']
        )

        manipulation_metrics = self.evaluate_manipulation_success(
            system.execute_manipulation_tasks(test_dataset['manipulation_tasks']),
            test_dataset['expected_outcomes']
        )

        report = f"""
        Language-Guided Manipulation Performance Report
        ==============================================

        Command Understanding:
        - Command Accuracy: {command_metrics['command_accuracy']:.3f}
        - Command Precision: {command_metrics['command_precision']:.3f}
        - Command Recall: {command_metrics['command_recall']:.3f}
        - Command F1-Score: {command_metrics['command_f1']:.3f}

        Object Grounding:
        - Grounding Accuracy: {grounding_metrics['grounding_accuracy']:.3f}
        - IoU Threshold: {grounding_metrics['iou_threshold']}

        Manipulation Success:
        - Task Success Rate: {manipulation_metrics['success_rate']:.3f}
        - Successful Tasks: {manipulation_metrics['successful_tasks']}/{manipulation_metrics['total_tasks']}

        System Assessment:
        """

        if command_metrics['command_f1'] > 0.8:
            report += "- Excellent command understanding\n"
        elif command_metrics['command_f1'] > 0.6:
            report += "- Good command understanding\n"
        else:
            report += "- Command understanding needs improvement\n"

        if grounding_metrics['grounding_accuracy'] > 0.7:
            report += "- Strong object grounding capabilities\n"
        elif grounding_metrics['grounding_accuracy'] > 0.5:
            report += "- Adequate object grounding\n"
        else:
            report += "- Object grounding needs improvement\n"

        if manipulation_metrics['success_rate'] > 0.8:
            report += "- High manipulation success rate\n"
        elif manipulation_metrics['success_rate'] > 0.6:
            report += "- Reasonable manipulation success\n"
        else:
            report += "- Manipulation performance needs improvement\n"

        return report
```

## Summary

Language-guided manipulation represents a significant advancement in human-robot interaction, enabling robots to understand and execute natural language commands for complex manipulation tasks. The key components include:

- **Command Processing**: Understanding and parsing natural language commands
- **Object Grounding**: Connecting language to specific visual objects
- **Action Planning**: Converting high-level commands to executable actions
- **Skill Execution**: Performing manipulation tasks using learned skills
- **Perception Integration**: Combining visual and linguistic information

The integration of these components creates a system capable of interpreting human instructions and executing corresponding manipulation behaviors, bridging the gap between natural language communication and robotic action.

In the next section, we'll explore how to integrate all these VLA components into a complete, end-to-end trainable system.