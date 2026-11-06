#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: Final Project
# Author: ECE 5377/7377
# Description: Occupancy Sensing Transmitter
# GNU Radio version: 3.10.12.0

from gnuradio import blocks
import pmt
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




class occupancy_sensing_transmitter(gr.top_block):

    def __init__(self):
        gr.top_block.__init__(self, "Final Project", catch_exceptions=True)
        self.flowgraph_started = threading.Event()

        ##################################################
        # Variables
        ##################################################
        self.lab_root = lab_root = os.environ.get('ECE7377_LAB5_ROOT', os.path.expanduser('~/ECE7377/OccupancySensing'))
        self.tx_file = tx_file = os.environ.get('ECE7377_TX_FILE', os.path.join(lab_root, 'work/waveforms/tx_pilot_file.dat'))
        self.tx_subdevice = tx_subdevice = os.environ.get('ECE7377_TX_SUBDEVICE', 'A:0')
        self.tx_gain = tx_gain = float(os.environ.get('ECE7377_TX_GAIN', '50'))
        self.tx_device_args = tx_device_args = os.environ.get('ECE7377_TX_DEVICE_ARGS', 'addr=10.61.40.60')
        self.tx_antenna = tx_antenna = os.environ.get('ECE7377_TX_ANTENNA', 'TX/RX')
        self.transmit_samples = transmit_samples = os.path.getsize(tx_file) // 8
        self.samp_rate = samp_rate = float(os.environ.get('ECE7377_SAMPLE_RATE', '1e6'))
        self.center_frequency = center_frequency = float(os.environ.get('ECE7377_CENTER_FREQUENCY', '2.45e9'))

        ##################################################
        # Blocks
        ##################################################

        self.uhd_usrp_sink_0 = uhd.usrp_sink(
            ",".join((tx_device_args, "", "master_clock_rate=125e6")),
            uhd.stream_args(
                cpu_format="fc32",
                otw_format="sc16",
                args='',
                channels=[0],
            ),
            "",
        )
        self.uhd_usrp_sink_0.set_clock_source('internal', 0)
        self.uhd_usrp_sink_0.set_time_source('internal', 0)
        self.uhd_usrp_sink_0.set_subdev_spec(tx_subdevice, 0)
        self.uhd_usrp_sink_0.set_samp_rate(samp_rate)
        # No synchronization enforced.

        self.uhd_usrp_sink_0.set_center_freq(center_frequency, 0)
        self.uhd_usrp_sink_0.set_antenna(tx_antenna, 0)
        self.uhd_usrp_sink_0.set_gain(tx_gain, 0)
        self.blocks_head_0 = blocks.head(gr.sizeof_gr_complex*1, transmit_samples)
        self.blocks_file_source_0 = blocks.file_source(gr.sizeof_gr_complex*1, tx_file, False, 0, 0)
        self.blocks_file_source_0.set_begin_tag(pmt.PMT_NIL)


        ##################################################
        # Connections
        ##################################################
        self.connect((self.blocks_file_source_0, 0), (self.blocks_head_0, 0))
        self.connect((self.blocks_head_0, 0), (self.uhd_usrp_sink_0, 0))


    def get_lab_root(self):
        return self.lab_root

    def set_lab_root(self, lab_root):
        self.lab_root = lab_root
        self.set_tx_file(os.environ.get('ECE7377_TX_FILE', os.path.join(self.lab_root, 'work/waveforms/tx_pilot_file.dat')))

    def get_tx_file(self):
        return self.tx_file

    def set_tx_file(self, tx_file):
        self.tx_file = tx_file
        self.set_transmit_samples(os.path.getsize(self.tx_file) // 8)
        self.blocks_file_source_0.open(self.tx_file, False)

    def get_tx_subdevice(self):
        return self.tx_subdevice

    def set_tx_subdevice(self, tx_subdevice):
        self.tx_subdevice = tx_subdevice

    def get_tx_gain(self):
        return self.tx_gain

    def set_tx_gain(self, tx_gain):
        self.tx_gain = tx_gain
        self.uhd_usrp_sink_0.set_gain(self.tx_gain, 0)

    def get_tx_device_args(self):
        return self.tx_device_args

    def set_tx_device_args(self, tx_device_args):
        self.tx_device_args = tx_device_args

    def get_tx_antenna(self):
        return self.tx_antenna

    def set_tx_antenna(self, tx_antenna):
        self.tx_antenna = tx_antenna
        self.uhd_usrp_sink_0.set_antenna(self.tx_antenna, 0)

    def get_transmit_samples(self):
        return self.transmit_samples

    def set_transmit_samples(self, transmit_samples):
        self.transmit_samples = transmit_samples
        self.blocks_head_0.set_length(self.transmit_samples)

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.uhd_usrp_sink_0.set_samp_rate(self.samp_rate)

    def get_center_frequency(self):
        return self.center_frequency

    def set_center_frequency(self, center_frequency):
        self.center_frequency = center_frequency
        self.uhd_usrp_sink_0.set_center_freq(self.center_frequency, 0)




def main(top_block_cls=occupancy_sensing_transmitter, options=None):
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
