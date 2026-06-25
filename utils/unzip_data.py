import py7zr

data_path = "/global/cfs/cdirs/m4633/puren/interp_dm/"
# dataset = Shanghai(data_path + '/shanghai_radar', 128)'
# /pscratch/sd/p/puren93/puren/Interp-DM

# sea-temperature.zip

with py7zr.SevenZipFile(data_path + 'z500-era5.zip', mode='r') as z:
    z.extractall(path=data_path + 'z500_era5')