from dpm_recovery import DPM_Recovery

if __name__ == '__main__':
    image = 'images/image_4.jpg'
    temp = 'temps/temp3.jpg'
    # image = 'images/image_2.jpg'
    # temp = 'temps/temp2.jpg'
    # image = 'images/image_1.jpg'
    # temp = 'temps/temp.jpg'
    dpm_recovery = DPM_Recovery(image, temp)
    dpm_recovery.binarization()
    dpm_recovery.find_point_groups()
    dpm_recovery.defining_code_boundaries()
    dpm_recovery.visualization()
