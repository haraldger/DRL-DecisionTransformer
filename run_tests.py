from tests import data_collection_test, resnet_test, dqn_test

def run():
    dqn_test.run()
    data_collection_test.run()
    resnet_test.run()



if __name__ == '__main__':
    run()