#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: Lab 3: N310 to Finite File Sink
# Author: ECE 5377/7377
# Description: Lab 3 finite complex64 receiver
# GNU Radio version: 3.10.12.0

from gnuradio import blocks
from gnuradio import gr
from gnuradio.filter import firdes
from gnuradio.fft import window
import sys
import signal
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
from gnuradio import uhd
import time
import os
import threading




class lab3_finite_receiver(gr.top_block):

    def __init__(self):
        gr.top_block.__init__(self, "Lab 3: N310 to Finite File Sink", catch_exceptions=True)
        self.flowgraph_started = threading.Event()

        ##################################################
        # Variables
        ##################################################
        self.lab_root = lab_root = os.environ.get('ECE7377_LAB3_ROOT', os.path.expanduser('~/ECE7377'))
        self.work_dir = work_dir = os.path.join(lab_root, 'work')
        self.samp_rate = samp_rate = 1.25e6
        self.capture_seconds = capture_seconds = float(os.environ.get('ECE7377_CAPTURE_SECONDS', '8.0'))
        self.rx_subdevice = rx_subdevice = os.environ.get('ECE7377_RX_SUBDEVICE', 'A:0')
        self.rx_gain = rx_gain = float(os.environ.get('ECE7377_RX_GAIN_DB', '60'))
        self.rx_device_args = rx_device_args = os.environ.get('ECE7377_RX_DEVICE_ARGS', 'addr=10.61.40.174')
        self.rx_antenna = rx_antenna = os.environ.get('ECE7377_RX_ANTENNA', 'RX2')
        self.output_file = output_file = os.environ.get('ECE7377_OUTPUT_FILE', os.path.join(work_dir, 'captures/task4_tone_c64.dat'))
        self.center_frequency = center_frequency = 2.45e9
        self.capture_samples = capture_samples = int(round(samp_rate * capture_seconds))

        ##################################################
        # Blocks
        ##################################################

        self.uhd_usrp_source_0 = uhd.usrp_source(
            ",".join((rx_device_args, "", "master_clock_rate=125e6")),
            uhd.stream_args(
                cpu_format="fc32",
                otw_format="sc16",
                args='',
                channels=[0],
            ),
        )
        self.uhd_usrp_source_0.set_clock_source('internal', 0)
        self.uhd_usrp_source_0.set_time_source('internal', 0)
        self.uhd_usrp_source_0.set_subdev_spec(rx_subdevice, 0)
        self.uhd_usrp_source_0.set_samp_rate(samp_rate)
        # No synchronization enforced.

        self.uhd_usrp_source_0.set_center_freq(center_frequency, 0)
        self.uhd_usrp_source_0.set_antenna(rx_antenna, 0)
        self.uhd_usrp_source_0.set_gain(rx_gain, 0)
        self.blocks_head_0 = blocks.head(gr.sizeof_gr_complex*1, capture_samples)
        self.blocks_file_sink_0 = blocks.file_sink(gr.sizeof_gr_complex*1, output_file, False)
        self.blocks_file_sink_0.set_unbuffered(False)


        ##################################################
        # Connections
        ##################################################
        self.connect((self.blocks_head_0, 0), (self.blocks_file_sink_0, 0))
        self.connect((self.uhd_usrp_source_0, 0), (self.blocks_head_0, 0))


    def get_lab_root(self):
        return self.lab_root

    def set_lab_root(self, lab_root):
        self.lab_root = lab_root
        self.set_work_dir(os.path.join(self.lab_root, 'work'))

    def get_work_dir(self):
        return self.work_dir

    def set_work_dir(self, work_dir):
        self.work_dir = work_dir
        self.set_output_file(os.environ.get('ECE7377_OUTPUT_FILE', os.path.join(self.work_dir, 'captures/task4_tone_c64.dat')))

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.set_capture_samples(int(round(self.samp_rate * self.capture_seconds)))
        self.uhd_usrp_source_0.set_samp_rate(self.samp_rate)

    def get_capture_seconds(self):
        return self.capture_seconds

    def set_capture_seconds(self, capture_seconds):
        self.capture_seconds = capture_seconds
        self.set_capture_samples(int(round(self.samp_rate * self.capture_seconds)))

    def get_rx_subdevice(self):
        return self.rx_subdevice

    def set_rx_subdevice(self, rx_subdevice):
        self.rx_subdevice = rx_subdevice

    def get_rx_gain(self):
        return self.rx_gain

    def set_rx_gain(self, rx_gain):
        self.rx_gain = rx_gain
        self.uhd_usrp_source_0.set_gain(self.rx_gain, 0)

    def get_rx_device_args(self):
        return self.rx_device_args

    def set_rx_device_args(self, rx_device_args):
        self.rx_device_args = rx_device_args

    def get_rx_antenna(self):
        return self.rx_antenna

    def set_rx_antenna(self, rx_antenna):
        self.rx_antenna = rx_antenna
        self.uhd_usrp_source_0.set_antenna(self.rx_antenna, 0)

    def get_output_file(self):
        return self.output_file

    def set_output_file(self, output_file):
        self.output_file = output_file
        self.blocks_file_sink_0.open(self.output_file)

    def get_center_frequency(self):
        return self.center_frequency

    def set_center_frequency(self, center_frequency):
        self.center_frequency = center_frequency
        self.uhd_usrp_source_0.set_center_freq(self.center_frequency, 0)

    def get_capture_samples(self):
        return self.capture_samples

    def set_capture_samples(self, capture_samples):
        self.capture_samples = capture_samples
        self.blocks_head_0.set_length(self.capture_samples)




def main(top_block_cls=lab3_finite_receiver, options=None):
    tb = top_block_cls()

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()

        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    tb.start()
    tb.flowgraph_started.set()

    tb.wait()


if __name__ == '__main__':
    main()
