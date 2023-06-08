import sys
import np_nwb.metadata.from_np_session as from_np_session

if __name__ == '__main__':
    from_np_session.main(*sys.argv[1:])
