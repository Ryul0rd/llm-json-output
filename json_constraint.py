from typing import List, Dict, Callable, Type, Union
from dataclasses import dataclass, fields
from abc import ABC, abstractmethod
import torch
from transformers import PreTrainedTokenizerBase

PrefixAllowedTokensFn = Callable[[int, torch.Tensor], List[int]]


@dataclass
class BatchedJsonConstraint:
    prompt_ids: torch.Tensor
    tokenizer: PreTrainedTokenizerBase
    schemas: Union[Type, List[Type]]

    def __post_init__(self):
        batch_size = self.prompt_ids.shape[0]
        if not isinstance(self.schemas, list):
            self.schemas = [self.schemas for _ in range(batch_size)]
        self.constraints = [JsonConstraint(self.prompt_ids[index], self.tokenizer, schema) for index, schema in enumerate(self.schemas)]

    def __call__(self, batch_index: int, input_ids: torch.Tensor) -> List[int]:
        json_constraint = self.constraints[batch_index]
        valid_token_ids = json_constraint(input_ids)
        return valid_token_ids


@dataclass
class JsonConstraint:
    prompt_ids: torch.Tensor
    tokenizer: PreTrainedTokenizerBase
    schema: Type

    def __post_init__(self):
        self.prompt_length = self.prompt_ids.shape[0]
        self.json_state_machine = JsonObjectStateMachine(self.schema)
        # cursed hack to get around the fact that a leading space on a token is represented with different characters in different tokenizers
        token_id_a = self.tokenizer.encode("a")[0]
        self.token_str_index_map = {self.tokenizer.convert_ids_to_tokens([token_id_a, index])[1]: index for _, index in self.tokenizer.get_vocab().items()}

    def __call__(self, input_ids: torch.Tensor) -> List[int]:
        """
        input_ids is a 1d tensor of type int
        """
        # figure out what has been generated so far
        if input_ids.shape[0] == self.prompt_length:
            generated_so_far = ""
        else:
            generated_so_far = self.tokenizer.decode(input_ids[self.prompt_length:])

        # traverse state machine using text generated so far
        self.json_state_machine.reset()
        self.json_state_machine.advance(generated_so_far)

        # for each token str in the vocab, check if it makes legal moves in the state machine from the current position
        allowed_token_ids = []
        for token_str, token_id in self.token_str_index_map.items():
            if self.json_state_machine.valid_continuation(token_str) and token_str != "":
                allowed_token_ids.append(token_id)

        if len(allowed_token_ids) == 0:
            allowed_token_ids.append(self.tokenizer.eos_token_id)
        
        return allowed_token_ids


# types to support:
# int,    DONE!
# float,  DONE* (no exponent support yet)
# bool,   DONE!
# str,    DONE* (no escape chars yet)
# list,   X
# option, X
# enum,   X
# obj     DONE!


class JsonStateMachineModule(ABC):
    @abstractmethod
    def advance_char(self, char: str) -> str:
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def save(self):
        pass

    @abstractmethod
    def load(self):
        pass


class JsonObjectStateMachine(JsonStateMachineModule):
    def __init__(self, schema: Type):
        self.state = 0
        self.backup_state = None
        self.test = 0
        self.field_dfas: Dict[str, JsonStateMachineModule] = {}
        self.states = self._build_states(schema)
        self.terminal_state = len(self.states) - 1

    def _build_states(self, schema: Type) -> List[Dict[str, int]]:
        states = ["start", "{"]

        for field in fields(schema):
            states.extend(list(f'"{field.name}":'))
            if field.type == str:
                states.append(f"str_{field.name}")
                self.field_dfas[f"str_{field.name}"] = JsonStrStateMachine()
            elif field.type == int:
                states.append(f"int_{field.name}")
                self.field_dfas[f"int_{field.name}"] = JsonIntStateMachine()
            elif field.type == float:
                states.append(f"float_{field.name}")
                self.field_dfas[f"float_{field.name}"] = JsonFloatStateMachine()
            elif field.type == bool:
                states.append(f"bool_{field.name}")
                self.field_dfas[f"bool_{field.name}"] = JsonBoolStateMachine()
            elif isinstance(field.type, Type):
                states.append(f"obj_{field.name}")
                self.field_dfas[f"obj_{field.name}"] = JsonObjectStateMachine(field.type)
            else:
                raise ValueError()
            states.append(",")
    
        if states[-1] == ",":
            states = states[:-1]
        states.append("}")

        return states

    def advance(self, s: str):
        for char in s:
            self.advance_char(char)

    def advance_char(self, char: str) -> str:
        """
        Returns 'advanced' if the state advanced,
        'finished' if we got an unexpected char in a terminal state,
        and 'error' if we got an unexpected char in a non-terminal state
        """
        # This entire method body is cancer
        if self.state != self.terminal_state:
            current_state = self.states[self.state]
            next_state = self.states[self.state+1]

        if self.state == self.terminal_state:
            return "finished"
        elif next_state in self.field_dfas:
            result = self.field_dfas[next_state].advance_char(char)
            if result == "advanced":
                self.state += 1
                return "advanced"
            else:
                return "error"
        elif current_state in self.field_dfas:
            result = self.field_dfas[current_state].advance_char(char)
            if result == "finished":
                if next_state == char:
                    self.state += 1
                    return "advanced"
                else:
                    return "error"
            else:
                return result
        elif next_state == char:
            self.state += 1
            return "advanced"
        else:
            return "error"

    def reset(self):
        self.state = 0
        for _, state_machine in self.field_dfas.items():
            state_machine.reset()

    def save(self):
        self.backup_state = self.state
        for _, state_machine in self.field_dfas.items():
            state_machine.save()

    def load(self):
        self.state = self.backup_state
        for _, state_machine in self.field_dfas.items():
            state_machine.load()

    def valid_continuation(self, s: str) -> bool:
        self.save()
        valid_so_far = True
        for char in s:
            if self.advance_char(char) != "advanced":
                valid_so_far = False
                break
        self.load()
        return valid_so_far


class JsonStrStateMachine(JsonStateMachineModule):
    def __init__(self):
        self.state = "start"
        self.dfa = {
            ("start", '"'): "open_quote",
            ("open_quote", '"'): "close_quote",
            ("open_quote", "wildcard"): "wildcard",
            ("wildcard", "wildcard"): "wildcard",
            ("wildcard", '"'): "close_quote",
        }
        self.terminal_state = 'close_quote'

    def advance_char(self, char: str) -> str:
        """
        Returns 'advanced' if the state advanced,
        'finished' if we got an unexpected char in a terminal state,
        and 'error' if we got an unexpected char in a non-terminal state
        """
        if char not in {'"', "\\"}:
            char = "wildcard"

        if (self.state, char) in self.dfa:
            self.state = self.dfa[(self.state, char)]
            return "advanced"
        elif self.state == self.terminal_state:
            return "finished"
        else:
            return "error"

    def reset(self):
        self.state = "start"

    def save(self):
        self.backup_state = self.state

    def load(self):
        self.state = self.backup_state


class JsonIntStateMachine(JsonStateMachineModule):
    def __init__(self):
        self.nonzero_digits = {str(i) for i in range(1, 10)}
        self.state = "start"
        self.dfa = {
            ("start", "-"): "-",
            ("start", "0"): "0",
            ("start", "nonzero_digit"): "nonzero_digit",
            ("-", "0"): "0",
            ("-", "nonzero_digit"): "nonzero_digit",
            ("nonzero_digit", "0"): "digit",
            ("nonzero_digit", "nonzero_digit"): "digit",
            ("digit", "0"): "digit",
            ("digit", "nonzero_digit"): "digit",
        }
        self.terminal_states = {"0", "nonzero_digit", "digit"}

    def advance_char(self, char: str) -> str:
        """
        Returns 'advanced' if the state advanced,
        'finished' if we got an unexpected char in a terminal state,
        and 'error' if we got an unexpected char in a non-terminal state
        """
        if char in self.nonzero_digits:
            char = "nonzero_digit"

        if (self.state, char) in self.dfa:
            self.state = self.dfa[(self.state, char)]
            return "advanced"
        elif self.state in self.terminal_states:
            return "finished"
        else:
            return "error"
    
    def reset(self):
        self.state = "start"

    def save(self):
        self.backup_state = self.state

    def load(self):
        self.state = self.backup_state


class JsonFloatStateMachine(JsonStateMachineModule):
    def __init__(self):
        self.nonzero_digits = {str(i) for i in range(1, 10)}
        self.state = "start"
        self.dfa = {
            ("start", "-"): "-",
            ("start", "0"): "0",
            ("start", "nonzero_digit"): "nonzero_digit",
            ("-", "0"): "0",
            ("-", "nonzero_digit"): "nonzero_digit",
            ("nonzero_digit", "0"): "digit",
            ("nonzero_digit", "nonzero_digit"): "digit",
            ("digit", "0"): "digit",
            ("digit", "nonzero_digit"): "digit",
            ("0", "."): ".",
            ("nonzero_digit", "."): ".",
            ("digit", "."): ".",
            (".", "nonzero_digit"): "post_decimal_digit",
            (".", "0"): "post_decimal_digit",
            ("post_decimal_digit", "nonzero_digit"): "post_decimal_digit",
            ("post_decimal_digit", "0"): "post_decimal_digit",
        }
        self.terminal_states = {"0", "nonzero_digit", "digit", "post_decimal_digit"}

    def advance_char(self, char: str) -> str:
        """
        Returns 'advanced' if the state advanced,
        'finished' if we got an unexpected char in a terminal state,
        and 'error' if we got an unexpected char in a non-terminal state
        """
        if char in self.nonzero_digits:
            char = "nonzero_digit"

        if (self.state, char) in self.dfa:
            self.state = self.dfa[(self.state, char)]
            return "advanced"
        elif self.state in self.terminal_states:
            return "finished"
        else:
            return "error"

    def reset(self):
        self.state = "start"

    def save(self):
        self.backup_state = self.state

    def load(self):
        self.state = self.backup_state


class JsonBoolStateMachine(JsonStateMachineModule):
    def __init__(self):
        self.state = "start"
        self.dfa = {
            ("start", "t"): "tt",
            ("tt", "r"): "tr",
            ("tr", "u"): "tu",
            ("tu", "e"): "te",
            ("start", "f"): "ff",
            ("ff", "a"): "fa",
            ("fa", "l"): "fl",
            ("fl", "s"): "fs",
            ("fs", "e"): "fe",     
        }
        self.terminal_states = {"te", "fe"}

    def advance_char(self, char: str) -> str:
        """
        Returns 'advanced' if the state advanced,
        'finished' if we got an unexpected char in a terminal state,
        and 'error' if we got an unexpected char in a non-terminal state
        """
        if (self.state, char) in self.dfa:
            self.state = self.dfa[(self.state, char)]
            return "advanced"
        elif self.state in self.terminal_states:
            return "finished"
        else:
            return "error"

    def reset(self):
        self.state = "start"

    def save(self):
        self.backup_state = self.state

    def load(self):
        self.state = self.backup_state
