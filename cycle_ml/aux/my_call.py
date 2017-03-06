
import subprocess

def my_call(arglst):
    """ A wrapper to check_call 
    reports exceptions as return codes
    """
    try:
        print(arglst)
        subprocess.check_call(arglst, stdout=1, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        return e.returncode
    except Exception as e:
        print("subporcess.check_call failed %s" % str(e))
        return 2
    return 0


