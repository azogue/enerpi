# -*- coding: utf-8 -*-
"""
ENERPI - CRON Scheduler

- Print User-CRON table
- Set command on reboot
- Set periodic command
- Remove CRON command

"""
from crontab import CronTab
import datetime as dt


def info_crontable():
    """
    Print current user CRON TABLE

    """
    ahora = dt.datetime.now()
    cron = CronTab(user=True)
    print('-*-' * 30)
    for job in cron:
        # Validity Check:
        assert(job.is_valid())
        schedule = job.schedule(date_from=ahora)
        datetime_n = schedule.get_next()
        print('** JOB: {}'.format(job))
        print('nº execs this year: {}'.format(job.frequency_per_year(year=ahora.year)))
        print('NEXT: {:%H:%M %d-%m-%Y}'.format(datetime_n))
        print('PREV: {:%H:%M %d-%m-%Y}'.format(schedule.get_prev()))
        print('*' * 90)
    print('-*-' * 30)


def set_command_on_reboot(command, comment=None, verbose=True):
    """
    Sets CRON task in current user crontab (like editing it with 'crontab -e') at @reboot
    :param command: :str: CRON command
    :param comment: :str: comment about the command
    :param verbose: :bool: shows info about the CRON job
    """
    cron = CronTab(user=True)
    for job in cron.find_command(command):
        cron.remove(job)
    if comment is None:
        new_job = cron.new(command=command)
    else:
        new_job = cron.new(command=command, comment=comment)
    new_job.every_reboot()
    if verbose:
        print('->CRON JOB: {}'.format(new_job))
        # print('job.is_valid = {}'.format(job.is_valid()))
    # Validity Check:
    assert(new_job.is_valid())
    # Write CronTab back to system or filename:
    cron.write()


def set_command_periodic(command, comment=None, hour=None, minute=None, frecuency_days=None, verbose=True):
    """
    Sets CRON task in current user crontab (like editing it with 'crontab -e') at some frequency
    :param command: :str: CRON command
    :param comment: :str: comment about the command
    :param hour: :int or None: every X hours
    :param minute: :int or None: every X minutes
    :param frecuency_days: :int or None: every X days
    :param verbose: :bool: shows info about the CRON job
    """
    cron = CronTab(user=True)
    for job in cron.find_command(command):
        cron.remove(job)
    if comment is None:
        new_job = cron.new(command=command)
    else:
        new_job = cron.new(command=command, comment=comment)
    if hour is not None:
        new_job.hour.every(hour)
    if minute is not None:
        new_job.minute.every(minute)
    if frecuency_days is not None:
        new_job.day.every(frecuency_days)
    assert(new_job.is_valid())

    schedule = new_job.schedule(date_from=dt.datetime.now())
    datetime_n = schedule.get_next()
    if verbose:
        print('->CRON JOB: {}\n  * NEXT PROGRAMMED EXEC: {:%H:%M %d-%m-%Y}'.format(new_job, datetime_n))
    # Write CronTab back to system or filename:
    cron.write()
    return datetime_n


def clear_cron_commands(commands, verbose=True):
    """
    Deletes a list of CRON tasks (as list of commands)
    :param commands: list of commands to delete
    :param verbose: :bool: shows info about the operation
    :return:
    """
    cron = CronTab(user=True)
    for command in commands:
        for job in cron.find_command(command):
            if verbose:
                print(' --> DELETING CRON-JOB: "{}"'.format(job))
            cron.remove(job)
    cron.write()
    return True


if __name__ == '__main__':
    info_crontable()
