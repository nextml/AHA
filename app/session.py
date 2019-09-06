import uuid
import time

SESH_EXP_TIME = 60*15 # 15 minutes

class Option:

    def __init__(self, caption, score, label):
        self.caption = caption
        self.score = score
        self.label = label

class Session:

    def __init__(self):
        self.id = uuid.uuid4()
        self.time_created = time.time()
        self.options = []

    def active(self):
        if time.time() - SESH_EXP_TIME > self.time_created or self.time_created == None:
            print('SESSION EXPIRED')
            return false
        return True

class SessionManager:

    def __init__(self):
        self.sessions = []

    def check_for_session_with_id(self, id):
        for session in self.sessions:
            if session.active() == True and str(session.id) == str(id):
                return True

        return False

    def validate_session(self, args):
        """
        Checks if a url has a valid session
        Returns true if this is true
        """
        id = None

        for key in args:
            value = args.get(key)
            if key == 'id':
                id = value

        if id == None:
            print("Session id not found")
            return False

        return self.check_for_session_with_id(id)

    def new_session(self):
        n = Session()
        self.sessions.append(n)
        return n

    def get_session(self, id):
        for session in self.sessions:
            if str(id) == str(session.id):
                return session
        return None

