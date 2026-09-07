import datetime

def what_time_is_it():
    now = datetime.datetime.now()
    year = str(now.year)
    month = str(now.month).zfill(2)
    day = str(now.day).zfill(2)
    hour = str(now.hour).zfill(2)
    minute = str(now.minute).zfill(2)
    second = str(now.second).zfill(2)
    content = "{}-{}{}-{}-{}-{}".format(year, month, day, hour, minute, second)
    return content




