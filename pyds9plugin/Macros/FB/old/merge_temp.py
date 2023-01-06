def FindTimeField(liste):
    timestamps = [
        "Datation GPS",
        "CreationTime_ISO8601",
        "date",
        "Date",
        "Time",
        "Date_TU",
        "UT Time",
        "Date_Time",
        "Time UT",
        "DATETIME",
    ]
    timestamps_final = (
        [field.upper() for field in timestamps]
        + [field.lower() for field in timestamps]
        + timestamps
    )
    try:
        timestamps_final.remove("date")
    except ValueError:
        pass
    for timename in timestamps_final:
        if timename in liste:  # table.colnames:
            timefield = timename
    try:
        print("Time field found : ", timefield)
        return timefield
    except UnboundLocalError as e:
        print(e)
        return "DATETIME"  # liste[0]  # table.colnames[0]


def give_value_from_time(
    cat,  # temp file
    date_time,
    date_time_field=None,
    timeformatImage=None,
    TimeFieldCat=None,
    timeformatCat=None,
    columns=None,
    timediff=0,
):
    import numpy as np

    def RetrieveTimeFormat(time):
        formats = [
            "%d/%m/%Y %H:%M:%S.%f",
            "%Y-%m-%d %H:%M:%S.%f",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S",
            "%m/%d/%Y %H:%M:%S",
            "%m/%d/%y %H:%M:%S",
        ]
        form = []
        for formati in formats:
            try:
                # print(time)
                datetime.datetime.strptime(time, formati)
                form.append(True)
            except ValueError:
                form.append(False)
        return formats[np.argmax(form)]

    if TimeFieldCat is None:
        TimeFieldCat = FindTimeField(cat.colnames)
    if timeformatCat is None:
        timeformatCat = RetrieveTimeFormat(cat[TimeFieldCat][0])
    if date_time_field is None:
        date_time_field = FindTimeField(date_time.colnames)
    # print(cat[TimeFieldCat])
    cat = cat[~np.ma.is_masked(cat[TimeFieldCat])]
    cat["timestamp"] = [
        datetime.datetime.strptime(d, timeformatCat) for d in cat[TimeFieldCat]
    ]  # .timestamp()
    if timeformatImage is None:
        timeformatImage = RetrieveTimeFormat(date_time[date_time_field][0])
    # print("%s: %s is %s"%(date_time_field,date_time[date_time_field][0], timeformatImage))
    # print("%s: %s is %s"%(TimeFieldCat,cat[TimeFieldCat][0], timeformatCat))

    if columns is None:
        columns = cat.colnames
        columns.remove(TimeFieldCat)
        columns.remove("timestamp")
    for i, column in enumerate(columns):
        date_time[column] = np.nan
    for j, line in enumerate(date_time):
        timestamp_image = datetime.datetime.strptime(
            date_time[date_time_field][j], timeformatImage
        )
        for i, column in enumerate(columns):
            mask = np.isfinite(cat[column])  # .mask
            try:
                temp = cat[column][mask][
                    np.argmin(
                        abs(
                            cat[mask]["timestamp"]
                            + datetime.timedelta(hours=timediff)
                            - timestamp_image
                        )
                    )
                ]
                # print(date_time[col][j])
                date_time[column][j] = temp
            except ValueError:
                pass
            # print(date_time[column])

        # print(temp, type(temp))
    return date_time
