import numpy as np



def path(mode, num,select_1_6):
    # mode -> 1: Samsung
    # mode -> 2: Asus
    # mode -> 3: Asus_Ipad
    # mode -> 4: Lenovo
    if mode == 1:
        device = 'Samsung'
    if mode == 2:
        device = 'Asus'
    if mode == 3:
        device = 'Asus_Ipad'
    if mode == 4:
        device = 'Lenovo'

    # num -> 1: route1
    # num -> 2: route2
    # num -> 3: route3
    # num -> 4: route4
    if num == 1:
        route = '1'
    if num == 2:
        route = '2'
    if num == 3:
        route = '3'
    if num == 4:
        route = '4'
    if num == 5:
        route = '5'
    if num == 6:
        route = '6'
    if num == 7:
        route = '7'
    if num == 8:
        route = '8'
    if num == 9:
        route = '9'
    if num == 10:
        route = '10'

    #OUTPUT
    if (num <9):
        output0 = 'D:/CCU/ANguyen/Environment/Museum/Processed_Data/Code/route_'+ route +'/route_' + route + '_20_' + device + '.csv'
        output1 = 'D:/CCU/ANguyen/Environment/Museum/Processed_Data/Code/route_'+ route +'/route_'+route + '_label_20.csv'
        output2 = 'Route_' + route
    else:
        output0 = 'D:/CCU/ANguyen/Environment/Museum/Processed_Data/Code/route_'+ route +'/Route_' + route + '_'+str(select_1_6)+'_' + device + '.csv'
        output1 = 'D:/CCU/ANguyen/Environment/Museum/Processed_Data/Code/route_'+ route +'/Route_'+route +'_'+str(select_1_6)+ '_label.csv'
        output2 = 'Route_' + route
    return output0, output1, output2
