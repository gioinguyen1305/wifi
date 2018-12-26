import numpy as np

home_path = 'D:/CCU/ANguyen/RSS_Indoor_localization/wei_yuan_thesis/ALL/ALL/Dataset'


def path(environment, mode, num):

    # environment -> 0: 5F(Samsung), 1: 5F(Multi-devices)
    if environment == 0:

        # mode -> 0: Training data, 1: Testing data
        if mode == 0:

            # num -> 0: whole data + close, 1: whole data + open, 2: line data + close, 3: line data + open
            if num == 0:
                output0 = home_path + '/5F_SS/NPY/Training/all/TrainingData_5F_SS_close.npy'
                output1 = home_path + '/5F_SS/NPY/Training/all/TrainingLabel_5F_SS_close.npy'
                output2 = home_path + '/5F_SS/NPY/Training/all/Training_matrix.npy'
                output3 = '/all/close'

                return output0, output1, output2, output3

            if num == 1:
                output0 = home_path + '/5F_SS/NPY/Training/all/TrainingData_5F_SS_open.npy'
                output1 = home_path + '/5F_SS/NPY/Training/all/TrainingLabel_5F_SS_open.npy'
                output2 = home_path + '/5F_SS/NPY/Training/all/Training_matrix.npy'
                output3 = '/all/open'

                return output0, output1, output2, output3

            if num == 2:
                output0 = home_path + '/5F_SS/NPY/Training/line/TrainingData_5F_SS_close_line.npy'
                output1 = home_path + '/5F_SS/NPY/Training/line/TrainingLabel_5F_SS_close_line.npy'
                output2 = home_path + '/5F_SS/NPY/Training/line/Training_matrix_line.npy'
                output3 = '/line/close'

                return output0, output1, output2, output3

            if num == 3:
                output0 = home_path + '/5F_SS/NPY/Training/line/TrainingData_5F_SS_open_line.npy'
                output1 = home_path + '/5F_SS/NPY/Training/line/TrainingLabel_5F_SS_open_line.npy'
                output2 = home_path + '/5F_SS/NPY/Training/line/Training_matrix_line.npy'
                output3 = '/line/open'

                return output0, output1, output2, output3

        if mode == 1:

            # num -> 0: whole data + close, 1: whole data + open, 2: line data + close, 3: line data + open
            if num == 0:
                output0 = home_path + '/5F_SS/NPY/Testing/all/TestingData_5F_SS_fast_close.npy'
                output1 = home_path + '/5F_SS/NPY/Testing/all/TestingData_5F_SS_slow_close.npy'
                output2 = home_path + '/5F_SS/NPY/Testing/all/TestingLabel_5F_SS_fast_close.npy'
                output3 = home_path + '/5F_SS/NPY/Testing/all/Testing_matrix.npy'
                output4 = '/all/close'

                return output0, output1, output2, output3, output4

            if num == 1:
                output0 = home_path + '/5F_SS/NPY/Testing/all/TestingData_5F_SS_fast_open.npy'
                output1 = home_path + '/5F_SS/NPY/Testing/all/TestingData_5F_SS_slow_open.npy'
                output2 = home_path + '/5F_SS/NPY/Testing/all/TestingLabel_5F_SS_fast_open.npy'
                output3 = home_path + '/5F_SS/NPY/Testing/all/Testing_matrix.npy'
                output4 = '/all/open'

                return output0, output1, output2, output3, output4

            if num == 2:
                output0 = home_path + '/5F_SS/NPY/Testing/line/TestingData_5F_SS_fast_close_line.npy'
                output1 = home_path + '/5F_SS/NPY/Testing/line/TestingData_5F_SS_slow_close_line.npy'
                output2 = home_path + '/5F_SS/NPY/Testing/line/TestingLabel_5F_SS_fast_close_line.npy'
                output3 = home_path + '/5F_SS/NPY/Testing/line/Testing_matrix_line.npy'
                output4 = '/line/close'

                return output0, output1, output2, output3, output4

            if num == 3:
                output0 = home_path + '/5F_SS/NPY/Testing/line/TestingData_5F_SS_fast_open_line.npy'
                output1 = home_path + '/5F_SS/NPY/Testing/line/TestingData_5F_SS_slow_open_line.npy'
                output2 = home_path + '/5F_SS/NPY/Testing/line/TestingLabel_5F_SS_fast_open_line.npy'
                output3 = home_path + '/5F_SS/NPY/Testing/line/Testing_matrix_line.npy'
                output4 = '/line/open'

                return output0, output1, output2, output3, output4

    if environment == 1:

        if mode == 0:

            # red_case: ASUS_T00G, red_nocase: ASUS_Z00AD, white_nocase: ASUS_Z00D

            output0 = home_path + '/5F_multi-devices/NPY/Training/TrainingData_red_case.npy'
            output1 = home_path + '/5F_multi-devices/NPY/Training/TrainingData_red_nocase.npy'
            output2 = home_path + '/5F_multi-devices/NPY/Training/TrainingData_white_nocase.npy'
            output3 = home_path + '/5F_multi-devices/NPY/Training/TrainingLabel.npy'
            output4 = home_path + '/5F_multi-devices/NPY/matrix.npy'

            return output0, output1, output2, output3, output4

        if mode == 1:

            output0 = home_path + '/5F_multi-devices/NPY/Testing/TestingData_red_case.npy'
            output1 = home_path + '/5F_multi-devices/NPY/Testing/TestingData_red_nocase.npy'
            output2 = home_path + '/5F_multi-devices/NPY/Testing/TestingData_white_nocase.npy'
            output3 = home_path + '/5F_multi-devices/NPY/Testing/TestingLabel.npy'
            output4 = home_path + '/5F_multi-devices/NPY/matrix.npy'

            return output0, output1, output2, output3, output4


