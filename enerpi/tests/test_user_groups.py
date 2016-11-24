# -*- coding: utf-8 -*-
from subprocess import check_output, STDOUT, CalledProcessError
import sys


def _checkoutput(cmd):
    code = 0
    try:
        out = str(check_output(cmd, stderr=STDOUT, universal_newlines=True))
    except CalledProcessError as e:
        code = e.returncode
        out = str(e.output)
    return code, out


def _get_user_groups(user='www-data', verbose=True):
    # pi : pi kmem sudo video spi
    # www-data : www-data video spi
    # staff procmod everyone localaccounts _appserverusr admin _appserveradm _lpadmin com.apple.access_screensharing
    # com.apple.access_ssh access_bpf _appstore _lpoperator _developer com.apple.access_ftp
    code, out = _checkoutput(['groups', '{}'.format(user)])
    if code == 0:
        grupos = out.split(':')[-1].split('\n')[0].split()
        if verbose:
            print('** GROUP MEMBERSHIPS FOR USER "{}":\n\t{}'.format(user, ', '.join(grupos)))
        return True, grupos
    else:
        if verbose:
            print('ERROR IN GROUP MEMBERSHIPS FOR USER "{}"'.format(user))
        return False, out


def _check_user_groups(user='www-data', desired_groups=('gpio', 'spi'), fix=False):
    # spi i2c gpio video kmem www-data
    existent_groups = _checkoutput(['groups'])[1].split()
    not_existent_groups = [g for g in desired_groups if g not in existent_groups]
    user_exists, user_groups = _get_user_groups(user)
    if not user_exists:
        print("ERROR: USER '{}' don't exist: {}".format(user, '\n'.join(user_groups.split('\n')[:-1])))
        return False
    if not_existent_groups:
        print('*** NOT EXISTENT GROUPS: "{}"'.format(not_existent_groups))
        if fix:
            pass
            # TODO Append groups
        else:
            return False
    groups_needed_for_user = [g for g in desired_groups if g not in user_groups]
    if len(groups_needed_for_user) == 0:
        return True
    print(groups_needed_for_user)
    if fix:
        # Try to addusergroup: sudo adduser {user} {group}
        for g in groups_needed_for_user:
            code, out = _checkoutput(['sudo', 'adduser', '{}'.format(user), '{}'.format(g)])
            if code != 0:
                print('ERROR: ADDING USER "{}" TO GROUP "{}"! -> {}'.format(user, g, out))
                return False
            print('* OK: ADDED USER "{}" TO GROUP "{}" -> {}'.format(user, g, out))
        return True
    return False


def test_user_groups():
    """
    Test for user & groups membership operations in linux environment
    """
    def _formatted_test(user_t, groups_t, fix=False):
        print('*' * 80 + '\nTEST (u="{}", g="{}", fix=False):'.format(user_t, groups_t))
        res = _check_user_groups(user=user_t, desired_groups=groups_t, fix=fix)
        print('********** TEST RESULT = {} **********'.format(res) + '\n' + '*' * 80)
        return res

    is_mac = sys.platform == 'darwin'
    users = ['www-data', 'uge', 'pi']
    groups_check = ['www-data', 'gpio', 'spi', 'kmem']
    print('IS macOS: {}'.format(is_mac))
    for u in users:
        _formatted_test(u, groups_check, fix=False)
    print('\n\n')

    # Correct for macOs (mine!)
    u = 'uge'
    groups_check = ['staff', 'everyone']
    result = _formatted_test(u, groups_check, fix=False)
    assert result == is_mac
    print('\n\n')

    # Appending main user to www-data
    u = 'pi'
    groups_check = ['www-data']
    _ = _formatted_test(u, groups_check, fix=False)
    # assert result == is_mac
    print('\n\n')

    # Appending www-data to needed groups
    u = 'www-data'
    groups_check = ['spi']
    _ = _formatted_test(u, groups_check, fix=False)
    # assert result == is_mac

    groups_check = ['gpio']
    _ = _formatted_test(u, groups_check, fix=False)
    # assert result == is_mac
    groups_check = ['kmem']
    _ = _formatted_test(u, groups_check, fix=False)
    # assert result == is_mac
    print('\n\n')

    user = 'www-data'
    needed_groups = [(('spi', ), ('gpio', 'kmem'))]
    print('\nCHECKING GROUPS MEMBERSHIP FOR USER "{}"'.format(user))
    groups_exist = _checkoutput(['groups'])[1].split()
    for need in needed_groups:
        print('need: {}'.format(need))
        for g in need:
            print('analysis g={}'.format(g))
            if g in groups_exist:
                print('{} in groups_exist'.format(g))
                result = _formatted_test(u, groups_check, fix=False)
                if result:
                    print('break')
                    break


def test_www_data_in_rpi():
    """
    Test for check user & groups membership needed to run ENERPI with:
        · The ENERPI logger daemon, running every reboot with CRON
        · The ENERPI flask webserver, running under NGINX + UWSGI-EMPEROR

    """
    # TODO check CUSTOM user & webserver user (from user CONFIG)
    def _formatted_test(user_t, groups_t, fix=False):
        print('*' * 80 + '\nTEST (u="{}", g="{}", fix=False):'.format(user_t, groups_t))
        res = _check_user_groups(user=user_t, desired_groups=groups_t, fix=fix)
        print('********** TEST RESULT = {} **********'.format(res) + '\n' + '*' * 80)
        return res

    if sys.platform == 'linux':
        user = 'www-data'
        needed_groups = [('spi', ), ('gpio', 'kmem')]
        print('\nCHECKING GROUPS MEMBERSHIP FOR USER "{}"'.format(user))
        groups_exist = _checkoutput(['groups'])[1].split()
        user_exists, user_groups = _get_user_groups(user)
        print('EXISTENT GROUPS: "{}"'.format(groups_exist))
        print('GROUPS WITH USER {} (exist={}): "{}"'.format(user, user_exists, user_groups))
        assert user_exists
        for need in needed_groups:
            print('need: {}'.format(need))
            for g in need:
                print('analysis g={}'.format(g))
                if g in groups_exist:
                    print('{} in groups_exist'.format(g))
                    result = _formatted_test(user, [g], fix=True)
                    print('user: {} in group: {}? {}--> '.format(user, g, result))
                    assert result
                    break


if __name__ == '__main__':
    test_user_groups()
    test_www_data_in_rpi()
