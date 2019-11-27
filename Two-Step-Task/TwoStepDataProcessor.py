from Network.DataProcessor import DataProcessor
from scipy.io import loadmat

class TwoStepDataProcessor(DataProcessor):

    def __init__(self, train_data_file, validate_data_file):
        super(TwoStepDataProcessor, self).__init__()
        self.train_data_file = train_data_file
        self.validate_data_file = validate_data_file
        #self.train_data_name = attrs['train_data_name'] # the key of training data
        #self.train_attr_name = attrs['train_brief_name'] # the key of training brief
        #self.train_guide_name = attrs['train_guide_name'] # the key for training guide in training brief
        #self.validate_data_name = attrs['validate_data_name']  # the key of training data
        #self.test_attr_name = attrs['train_brief_name']  # the key of training brief

    def prepareTrainingData(self):
        mat = loadmat(self.train_data_file)
        self.train_data_set = [t[0] for t in mat['data_ST']]
        self.train_guide = mat['data_ST_Brief']['training_guide'][0][0]
        return self.train_data_set, self.train_guide

    def prepareValidatingData(self):
        mat = loadmat(self.validate_data_file)
        self.validate_data_set = [t[0] for t in mat['data_ST']]
        self.validate_data_attr = mat['data_ST_Brief']
        return self.validate_data_set, self.validate_data_attr


if __name__ == '__main__':
    p = TwoStepDataProcessor('./data/SimpTwo_TrainingSet-2019_11_19-1.mat',
                             './data/SimpTwo_TestingSet-2019_11_19-1.mat',
                            )
    train_set, train_guide = p.prepareTrainingData()
    validate_set, validate_attr = p.prepareValidatingData()
    print("The size of a training trial is:", train_set[0].shape)
    print("The size of a train guide is:", train_guide.shape)
    print("The size of a validating trial is:", validate_set[0].shape)
