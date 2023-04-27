class Patien():
    def __init__(self, name, dob):
        'date of birth must be presented in dd/mm/yyyy format'
        self._name = name
        self._dob = dob

    def __str__(self):
        return "Patient: "  + self._name + ", born: " + self._dob

class Doctor():
    def __init__(self, name, department):
        self._name = name
        self._department = department

class Appointment():
    def __init__(self, date, time, patient, doctor, duration):
        self._date = date
        self._time = time
        self._patient = patient
        self._doctor = doctor
        self._duration = duration