# -*- coding: utf-8 -*-
from cryptography.fernet import Fernet, InvalidToken
from binascii import Error
import os
import socket
import sys
from time import time
from enerpi.base import DATA_PATH, CONFIG, log


# LAN broadcasting
UDP_IP = CONFIG.get('BROADCAST', 'UDP_IP', fallback="192.168.1.255")
UDP_PORT = CONFIG.getint('BROADCAST', 'UDP_PORT', fallback=57775)
DESCRIPTION_IO = "\tSENDER - RECEIVER vía UDP. Broadcast IP: {}, PORT: {}".format(UDP_IP, UDP_PORT)
KEY_FILE = os.path.join(DATA_PATH, CONFIG.get('BROADCAST', 'KEY_FILE', fallback='.secret_key'))


# TODO Automatize get encryption key (for tests)?
def get_encryption_key(key_file):
    """
    Encryption key for symmetric encryption. URL-safe base64-encoded 32-byte key

    * Get local key, or exec wizard for making one.

    :param: key_file: :str: path of the file with the encryption key
    :return: :bytes: encryption_key
    """

    try:
        secret_key = open(key_file).read().encode()
    except FileNotFoundError:
        print('\033[0m\033[1m\033[34mCrypto KEY is missing.\033[0m Please, select one option:\n'
              '\t\033[1m1. Generate a new key.\n\t2. Input an existent key. (Copy&paste from another computer)')
        selec = input('\033[34mOPTION: \033[0m\033[31m')
        if selec == '1':
            secret_key = Fernet.generate_key()
        else:
            secret_key = input('*** KEY: ').encode()
        if len(secret_key) > 10:
            print('\033[0mThe new KEY ("{}")\nwill be saved in "{}"\n'.format(secret_key, key_file))
            open(key_file, 'wb').write(secret_key)
        else:
            print('\033[31;1mNot a valid KEY!:\n"{}". Try again... BYE!\033[0m'.format(secret_key))
            sys.exit(-1)
    return secret_key


def get_codec(secret_key):
    """
    Get Fernet CODEC from secret_key with an AssertionError if Fernet error raises.

    :param secret_key: :bytes: encryption_key
    :return: Fernet()

    """
    # secret_key = get_encryption_key(key_file)
    try:
        codec = Fernet(secret_key)
        return codec
    except Error as error_fernet:
        log('Crypto KEY is not a valid KEY! -> {}.\nKEY="{}". Try again... BYE!'
            .format(error_fernet, secret_key), 'error', True)
        assert 0, 'Crypto KEY is not a valid KEY: {}'.format(error_fernet)
        # sys.exit(-1)


CODEC = get_codec(get_encryption_key(KEY_FILE))


def receiver_msg_generator(verbose=True, n_msgs=None, port=UDP_PORT, codec=CODEC):
    """
    Generador de mensajes en el receptor de la emisión en la broadcast IP. Recibe los mensajes y los desencripta.
    También devuelve los tiempos implicados en la recepción (~ el ∆T entre mensajes) y el proceso de desencriptado.
    :param verbose: :bool: Imprime Broadcast IP & PORT.
    :param n_msgs: :int: # de mensajes a recibir (ilimitados por defecto).
    :param port: :int: # de puerto de escucha.
    :param codec: :Fernet obj: Fernet object for decrypting msgs.
    :yield: msg, ∆T_msg, ∆T_decryp
    """
    sock, counter = None, 0
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((UDP_IP, port))
        log(DESCRIPTION_IO, 'ok', verbose, False)
        while (n_msgs is None) or (counter < n_msgs):
            tic = time()
            data, addr = sock.recvfrom(1024)  # buffer size is 1024 bytes
            toc_msg = time()
            try:
                msg = codec.decrypt(data).decode()
            except InvalidToken as e:
                log('InvalidToken ERROR: {}. La clave es correcta?\n* KEY: "{}"'
                    .format(e, get_encryption_key(key_file=KEY_FILE)), 'error', verbose, True)
                break
            toc_dcry = time()
            yield msg, toc_msg - tic, toc_dcry - toc_msg
            counter += 1
        # log('Closing receiver_msg_generator socket...', 'debug', verbose, True)
    # except OSError as e:
    #     log('OSError {} en receiver_msg_generator'.format(e), 'error', verbose, True)
    except KeyboardInterrupt:
        log('KeyboardInterrupt en receiver_msg_generator', 'warn', verbose, True)
    if sock is not None:
        sock.close()
    raise StopIteration


def broadcast_msg(msg, counter_unreachable, sock_send=None, codec=CODEC, verbose=True):
    """
    Emisión de datos en modo broadcast UDP (para múltiples receptores) como mensaje de texto encriptado.
    :param msg: Cadena de texto a enviar.
    :param counter_unreachable: np.array([0, 0]) de control de emisiones incorrectas (seguidas y totales)
    :param sock_send: Socket de envío broadcast. Se devuelve para su reutilización.
    :param codec: :Fernet obj: Fernet object for encrypting msgs.
    :param verbose: Imprime en stout mensajes de error y de envío de datos
    :return: sock_send
    """

    def _get_broadcast_socket(verb=False):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        # sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 32)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        log(DESCRIPTION_IO, 'ok', verb, verb)
        return sock

    if sock_send is None:
        sock_send = _get_broadcast_socket()
    encrypted_msg_b = codec.encrypt(msg.encode())
    try:
        sock_send.sendto(encrypted_msg_b, (UDP_IP, UDP_PORT))
        counter_unreachable[0] = 0
    except OSError as e:  # [Errno 101] Network is unreachable
        log('OSError: {}; C_UNREACHABLE: {}'.format(e, counter_unreachable), 'warn', verbose)
        counter_unreachable += 1
        sock_send = None
    # except Exception as e:
    #     log('ERROR en sendto: {} [{}]'.format(e, e.__class__), 'err', verbose)
    #     sock_send = _get_broadcast_socket()
    #     sock_send.sendto(encrypted_msg_b, (UDP_IP, UDP_PORT))
    log('SENDED: {}'.format(msg), 'debug', verbose, False)
    return sock_send
