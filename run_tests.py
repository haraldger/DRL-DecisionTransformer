from tests import data_collection_test, resnet_test, dqn_test, dt_agent_test

def run():
    dt_agent_test.run()
    data_collection_test.run()
    resnet_test.run()
    dqn_test.run()

if __name__ == '__main__':
    run()