class DefaultConfigs_C(object):
  src1_data = 'oulu'
  src1_train_num_frames = 1
  src2_data = 'replay'
  src2_train_num_frames = 1
  src3_data = 'msu'
  src3_train_num_frames = 1
  tgt_data = 'casia'
  tgt_test_num_frames = 2


class DefaultConfigs_I(object):
  src1_data = 'oulu'
  src1_train_num_frames = 1
  src2_data = 'casia'
  src2_train_num_frames = 1
  src3_data = 'msu'
  src3_train_num_frames = 1
  tgt_data = 'replay'
  tgt_test_num_frames = 2


class DefaultConfigs_M(object):
  src1_data = 'oulu'
  src1_train_num_frames = 1
  src2_data = 'casia'
  src2_train_num_frames = 1
  src3_data = 'replay'
  src3_train_num_frames = 1
  tgt_data = 'msu'
  tgt_test_num_frames = 2

class DefaultConfigs_O(object):
  src1_data = 'replay'
  src1_train_num_frames = 1
  src2_data = 'casia'
  src2_train_num_frames = 1
  src3_data = 'msu'
  src3_train_num_frames = 1
  tgt_data = 'oulu'
  tgt_test_num_frames = 2

configC = DefaultConfigs_C()
configI = DefaultConfigs_I()
configM = DefaultConfigs_M()
configO = DefaultConfigs_O()

