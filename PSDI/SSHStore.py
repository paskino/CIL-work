import zarr
import brem
import paramiko
import socket
import stat
import os
# from brem import remotepath
import posixpath
import pysnooper
import json

class SSHStore(zarr.DirectoryStore):
    def __init__(self, path, port=22, host=None,username=None,\
        private_key=None, remote_os='POSIX', logfile='ssh.log'):
        self.path = path
        self.cpars = {'host':host, 'port':port, 'username':username, 
                      'private_key':private_key, 'remote_os':remote_os, 
                      'logfile':logfile}

        

    def _connect(self):

        conn = brem.BasicRemoteExecutionManager(host=self.cpars['host'],
                       username=self.cpars['username'], port=self.cpars['port'],
                       private_key=self.cpars['private_key'], remote_os=self.cpars['remote_os'])
        
        conn.login(passphrase=False)
        
        return conn

    # @staticmethod
    # def _keys_fast(path, walker=os.walk):
        # for dirpath, _, filenames in walker(path):
        #     dirpath = os.path.relpath(dirpath, path)
        #     if dirpath == os.curdir:
        #         for f in filenames:
        #             yield f
        #     else:
        #         dirpath = dirpath.replace("\\", "/")
        #         for f in filenames:
        #             yield "/".join((dirpath, f))
    
    @pysnooper.snoop()
    def list_remote_zarr(self, path, keys=None, conn=None):
        if conn is None:
            conn = self._connect()
        curdir, fname, fattr = conn.listdir(path)
        if curdir is None:
            curdir = '.'
        if keys is None:
            keys = []
        for i,_ in enumerate(fname):
            kind = stat.S_IFMT(fattr[i].st_mode)
            if kind == stat.S_IFDIR:
                # a directory, do a recursive call
                self.list_remote_zarr(posixpath.join(path, curdir, fname[i]), keys, conn=conn )
            elif kind == stat.S_IFREG:
                # a file
                # keys.append( fname[i] )
                remote_fpath = posixpath.normpath(posixpath.join(path, curdir, fname[i]) )
                keys.append( remote_fpath )
                if fname[i] == '.zarray':
                    basename = posixpath.dirname(remote_fpath)
                    conn.changedir(basename)
                    conn.get_file(fname[i], localdir=os.getcwd())
                    with open(os.path.join(os.getcwd(), fname[i]), 'r') as jsf:
                        js = json.load(jsf)    
                        # js = zarr.util.json_loads(jsf_txt)
                    
        conn.logout()
        return keys

    def __getitem__(self, key):
        key = self._normalize_key(key)
        filepath = brem.remotepath.join(self.path, key)
        if brem.remotepath.isfile(filepath):
            return self._fromfile(filepath)
        else:
            raise KeyError(key)

if __name__ == '__main__':
    store = zarr.DirectoryStore('data/group.zarr')
    root = zarr.group(store=store, overwrite=True)
    # foo = root.create_group('foo')
    bar = root.zeros('bar', shape=(10, 10), chunks=(5, 5))
    bar[...] = 42
    
    store.close()

    # sstore = SSHStore('/home/edo/scratch/Data/PSDI/data/group.zarr', 
    #           host='vishighmem01.esc.rl.ac.uk', username='edo', 
    #           private_key=os.path.abspath('C:/Users/ofn77899/.ssh/id_rsa'))
    # keys = sstore.list_remote_zarr('/home/edo/scratch/Data/PSDI/data/group.zarr')
    # print (keys)


    store = zarr.storage.KVStore(dict())
    print (store.keys())