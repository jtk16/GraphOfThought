import gymnasium as gym
from gymnasium import spaces
import numpy as np

class MathEnv(gym.Env):
    def __init__(self, vocab, rev_vocab):
        super(MathEnv, self).__init__()
        self.vocab = vocab
        self.rev_vocab = rev_vocab
        self.action_space = spaces.Discrete(len(vocab)) # Action is to predict a token
        self.observation_space = spaces.Box(low=0, high=len(vocab)-1, shape=(20,), dtype=np.int32) # Max sequence length 20
        self.current_question = None
        self.current_answer_tokens = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        a = np.random.randint(0, 5)
        b = np.random.randint(0, 5)
        op_idx = np.random.randint(0, 4)
        ops = ['+', '-', '*', '/']
        op = ops[op_idx]

        self.current_question = f"{a}{op}{b}"
        self.current_answer_tokens = []

        # Initial observation is the tokenized question
        obs = self._tokenize_question(self.current_question)
        info = {}
        return obs, info

    def step(self, action):
        # Action is a token ID
        predicted_char = self.rev_vocab[action]
        self.current_answer_tokens.append(predicted_char)

        # Check if we have a complete answer (e.g., a number or 'inf')
        # This is a very simplified termination condition
        done = False
        reward = 0.0
        
        predicted_answer_str = "".join(self.current_answer_tokens)

        if len(self.current_answer_tokens) >= 1 and predicted_answer_str.strip() != '':
            try:
                true_answer = eval(self.current_question)
                predicted_answer = int(predicted_answer_str)
                if predicted_answer == true_answer:
                    reward = 1.0 # Reward for correct answer
                    done = True
                elif len(self.current_answer_tokens) >= 5: # Max answer length
                    done = True # Terminate if too long and not correct
            except (SyntaxError, ValueError, ZeroDivisionError):
                if predicted_answer_str == 'inf' and '/' in self.current_question and eval(self.current_question.split('/')[1]) == 0:
                    reward = 1.0
                    done = True
                elif len(self.current_answer_tokens) >= 5:
                    done = True
                else:
                    reward = -0.1 # Penalty for invalid answer or too long

        obs = self._tokenize_question(self.current_question) # Observation remains the question
        info = {}
        return obs, reward, done, False, info # obs, reward, terminated, truncated, info

    def _tokenize_question(self, question_str):
        token_ids = [self.vocab[char] for char in question_str if char in self.vocab]
        # Pad to max_seq_length
        padded_token_ids = token_ids + [self.vocab['<pad>']] * (self.observation_space.shape[0] - len(token_ids))
        return np.array(padded_token_ids, dtype=np.int32)
