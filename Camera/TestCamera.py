import vmbpy

vmb = vmbpy.VmbSystem.get_instance()
print('Starting')
with vmb:
    # cams = vmb.get_all_cameras()
    # for cam in cams:
    #     print(cam)
    print('test')
print('Done')