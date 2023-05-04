from tests import data_collection_test, resnet_test, dqn_test

def run():
    data_collection_test.run()
    resnet_test.run()
    dqn_test.run()


if __name__ == '__main__':
    run()