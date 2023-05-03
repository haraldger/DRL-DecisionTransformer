from Agents import dt_agent
import gym 
import torch
from Agents.dt_agent import DTAgent

def test_forward_pass():
    env = gym.make('ALE/MsPacman-v5')

    dt_model = DTAgent(env)

    # expects state to be float normalized form
    state_seq = torch.rand(2, 3, 210, 160).float()
    action_seq = torch.randint(high=9, size=(2,)).unsqueeze(-1)
    ret_to_go_seq = torch.tensor([10000,9998]).unsqueeze(-1).float()
    timestep_seq = torch.tensor([0,1]).unsqueeze(-1)

    # print(action_seq.shape)
    # print(ret_to_go_seq.shape)
    # print(timestep_seq.shape)

    next_action = dt_model.predict_next_action(state_seq, action_seq, ret_to_go_seq, timestep_seq)
    print("Next action prediction test: ", next_action)


def run():
    print("Testing DT Agent forward pass.\n")
    test_forward_pass()
    print("\n")