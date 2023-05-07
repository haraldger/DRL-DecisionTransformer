from tests import data_collection_test, dqn_test, bitchnet_test, dt_agent_test

def run():
    bitchnet_test.run()
    data_collection_test.run()
    dt_agent_test.run()
    dqn_test.run()

if __name__ == '__main__':
    run()