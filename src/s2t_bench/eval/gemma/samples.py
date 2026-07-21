"""Telecom field-report evaluation samples.

Each sample carries three things:
  raw_text       - noisy, ASR-style transcript (input to the correction task)
  corrected_text - the expected clean transcript (target for correction scoring)
  expected       - the expected structured fault report (target for JSON scoring)

These are hand-written XGS-PON / GPON fibre-access scenarios. Extend freely; the
scorers only rely on the field names in schema.py.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EvalSample:
    id: str
    raw_text: str
    corrected_text: str
    expected: dict


SAMPLES: list[EvalSample] = [
    EvalSample(
        id="flt-1001",
        raw_text="fault ten oh one, custmer had no service, the o n t was showing red los light, swapped the o n t for a new one sku ont dash five six seven eight and service came back, old unit was faulty",
        corrected_text="Fault 1001, customer had no service. The ONT was showing a red LOS light. Swapped the ONT for a new one, SKU ONT-5678, and service came back. Old unit was faulty.",
        expected={
            "fault_reference": "1001",
            "reported_issue_summary": "Customer had no service; ONT showing red LOS light.",
            "reported_issue_correct_flag": True,
            "valid_issue_flag": True,
            "rc_hl_category": "Hardware",
            "rc_ll_category": "ONT Failure",
            "rc_fault_story_verbose": "The customer ONT displayed a red LOS alarm indicating loss of optical signal at the unit. The ONT was replaced with SKU ONT-5678 and service was restored, confirming the original ONT had failed.",
            "faulty_components": [{"item": "ONT", "sku": "ONT-5678"}],
            "used_components": [{"item": "ONT", "sku": "ONT-5678"}],
        },
    ),
    EvalSample(
        id="flt-1002",
        raw_text="ticket one thousand two, excess pon o l t port was down, checked the s f p plus on line card and it was dead, replaced sfp sku sfp dash p dash ten g, port came up",
        corrected_text="Ticket 1002, XGS-PON OLT port was down. Checked the SFP+ on the line card and it was dead. Replaced SFP, SKU SFP-P-10G, port came up.",
        expected={
            "fault_reference": "1002",
            "reported_issue_summary": "XGS-PON OLT port down due to failed SFP+ on the line card.",
            "reported_issue_correct_flag": True,
            "valid_issue_flag": True,
            "rc_hl_category": "Hardware",
            "rc_ll_category": "SFP Fault",
            "rc_fault_story_verbose": "The XGS-PON OLT port was reported down. The SFP+ transceiver on the line card was found to be dead and was replaced with SKU SFP-P-10G, after which the port came back up.",
            "faulty_components": [{"item": "SFP+", "sku": "SFP-P-10G"}],
            "used_components": [{"item": "SFP+", "sku": "SFP-P-10G"}],
        },
    ),
    EvalSample(
        id="flt-1003",
        raw_text="job one oh oh three, customer reported slow speeds but line tested fine, optical power was good at minus eighteen d b m, no fault found, likely wifi issue on customer side",
        corrected_text="Job 1003, customer reported slow speeds but line tested fine. Optical power was good at -18 dBm. No fault found, likely WiFi issue on customer side.",
        expected={
            "fault_reference": "1003",
            "reported_issue_summary": "Customer reported slow speeds; line and optical power tested normal.",
            "reported_issue_correct_flag": False,
            "valid_issue_flag": False,
            "rc_hl_category": "Network",
            "rc_ll_category": "No Fault Found",
            "rc_fault_story_verbose": "The customer reported slow speeds, but the line tested fine with optical power at -18 dBm within budget. No network fault was found; the issue is likely with the customer's WiFi.",
            "faulty_components": [],
            "used_components": [],
        },
    ),
    EvalSample(
        id="flt-1004",
        raw_text="fault one thousand four there was a fibre break in the feeder between cabinet twelve and the splitter, spliced the fibre and reused existing pigtail sku pig dash s c dash apc, service restored to forty homes",
        corrected_text="Fault 1004, there was a fibre break in the feeder between cabinet 12 and the splitter. Spliced the fibre and reused existing pigtail, SKU PIG-SC-APC. Service restored to 40 homes.",
        expected={
            "fault_reference": "1004",
            "reported_issue_summary": "Fibre break in the feeder between cabinet 12 and the splitter affecting 40 homes.",
            "reported_issue_correct_flag": True,
            "valid_issue_flag": True,
            "rc_hl_category": "Physical Plant",
            "rc_ll_category": "Fibre Break",
            "rc_fault_story_verbose": "A fibre break was located in the feeder cable between cabinet 12 and the splitter, causing loss of service to 40 homes. The fibre was spliced and an existing pigtail SKU PIG-SC-APC was reused, restoring service.",
            "faulty_components": [{"item": "Feeder fibre", "sku": "PIG-SC-APC"}],
            "used_components": [{"item": "Pigtail", "sku": "PIG-SC-APC"}],
        },
    ),
    EvalSample(
        id="flt-1005",
        raw_text="ticket ten oh five o n t kept dropping every few minutes, turned out to be a firmware mismatch, upgraded o n t firmware to version three point two, stable since",
        corrected_text="Ticket 1005, ONT kept dropping every few minutes. Turned out to be a firmware mismatch. Upgraded ONT firmware to version 3.2, stable since.",
        expected={
            "fault_reference": "1005",
            "reported_issue_summary": "ONT dropping every few minutes due to firmware mismatch.",
            "reported_issue_correct_flag": True,
            "valid_issue_flag": True,
            "rc_hl_category": "Software",
            "rc_ll_category": "Firmware Mismatch",
            "rc_fault_story_verbose": "The ONT was dropping connection every few minutes. The root cause was a firmware mismatch between the ONT and the OLT. The ONT firmware was upgraded to version 3.2 and the link has been stable since.",
            "faulty_components": [],
            "used_components": [],
        },
    ),
    EvalSample(
        id="flt-1006",
        raw_text="fault one thousand six, high attenuation on the drop, measured minus twenty nine d b m at the o n t, found a dirty connector, cleaned s c apc connector and re seated, power back to minus twenty two",
        corrected_text="Fault 1006, high attenuation on the drop. Measured -29 dBm at the ONT. Found a dirty connector, cleaned SC/APC connector and re-seated. Power back to -22 dBm.",
        expected={
            "fault_reference": "1006",
            "reported_issue_summary": "High attenuation on the drop caused by a dirty SC/APC connector.",
            "reported_issue_correct_flag": True,
            "valid_issue_flag": True,
            "rc_hl_category": "Physical Plant",
            "rc_ll_category": "Optical Budget",
            "rc_fault_story_verbose": "Optical power at the ONT was measured at -29 dBm, indicating high attenuation on the drop. A dirty SC/APC connector was found; it was cleaned and re-seated, bringing the power back to -22 dBm within budget.",
            "faulty_components": [{"item": "SC/APC connector", "sku": "CON-SC-APC"}],
            "used_components": [],
        },
    ),
    EvalSample(
        id="flt-1007",
        raw_text="job ten oh seven wrong vlan was provisioned on the o n t so customer got no internet, corrected the vlan config to five hundred, service working",
        corrected_text="Job 1007, wrong VLAN was provisioned on the ONT so customer got no internet. Corrected the VLAN config to 500, service working.",
        expected={
            "fault_reference": "1007",
            "reported_issue_summary": "Customer had no internet due to wrong VLAN provisioned on the ONT.",
            "reported_issue_correct_flag": True,
            "valid_issue_flag": True,
            "rc_hl_category": "Configuration",
            "rc_ll_category": "Config Error",
            "rc_fault_story_verbose": "The customer had no internet access. The ONT had been provisioned with the wrong VLAN. The VLAN configuration was corrected to 500 and service began working.",
            "faulty_components": [],
            "used_components": [],
        },
    ),
    EvalSample(
        id="flt-1008",
        raw_text="fault one oh oh eight, the one to thirty two splitter in the street cabinet was cracked, water ingress, replaced splitter sku spl dash one three two, restored twenty subscribers",
        corrected_text="Fault 1008, the 1:32 splitter in the street cabinet was cracked with water ingress. Replaced splitter, SKU SPL-1-32, restored 20 subscribers.",
        expected={
            "fault_reference": "1008",
            "reported_issue_summary": "Cracked 1:32 splitter with water ingress in the street cabinet affecting 20 subscribers.",
            "reported_issue_correct_flag": True,
            "valid_issue_flag": True,
            "rc_hl_category": "Physical Plant",
            "rc_ll_category": "Splitter Fault",
            "rc_fault_story_verbose": "The 1:32 optical splitter in the street cabinet was found cracked with water ingress, causing loss of service to 20 subscribers. The splitter was replaced with SKU SPL-1-32 and service was restored.",
            "faulty_components": [{"item": "1:32 splitter", "sku": "SPL-1-32"}],
            "used_components": [{"item": "1:32 splitter", "sku": "SPL-1-32"}],
        },
    ),
    EvalSample(
        id="flt-1009",
        raw_text="ticket one thousand nine power outage at the exchange took the o l t offline, u p s failed to hold, replaced ups battery sku bat dash forty eight v, o l t back online",
        corrected_text="Ticket 1009, power outage at the exchange took the OLT offline. UPS failed to hold. Replaced UPS battery, SKU BAT-48V, OLT back online.",
        expected={
            "fault_reference": "1009",
            "reported_issue_summary": "OLT offline after exchange power outage; UPS failed to hold.",
            "reported_issue_correct_flag": True,
            "valid_issue_flag": True,
            "rc_hl_category": "Power",
            "rc_ll_category": "UPS Failure",
            "rc_fault_story_verbose": "A power outage at the exchange took the OLT offline because the UPS failed to hold the load. The failed UPS battery was replaced with SKU BAT-48V and the OLT came back online.",
            "faulty_components": [{"item": "UPS battery", "sku": "BAT-48V"}],
            "used_components": [{"item": "UPS battery", "sku": "BAT-48V"}],
        },
    ),
    EvalSample(
        id="flt-1010",
        raw_text="fault ten ten customer complained of no service but appointment was a no access, could not get into the property, rebooked, no work done",
        corrected_text="Fault 1010, customer complained of no service but appointment was a no-access. Could not get into the property, rebooked, no work done.",
        expected={
            "fault_reference": "1010",
            "reported_issue_summary": "Reported no service but engineer could not access the property.",
            "reported_issue_correct_flag": False,
            "valid_issue_flag": True,
            "rc_hl_category": "Physical Plant",
            "rc_ll_category": "No Access",
            "rc_fault_story_verbose": "The customer reported no service. On attendance the engineer could not gain access to the property, so no diagnosis or work was carried out. The appointment was rebooked.",
            "faulty_components": [],
            "used_components": [],
        },
    ),
    EvalSample(
        id="flt-1011",
        raw_text="job one oh one one, uplink card in the o l t chassis was faulting with c r c errors, swapped the line card sku lc dash xgs dash sixteen, errors cleared",
        corrected_text="Job 1011, uplink card in the OLT chassis was faulting with CRC errors. Swapped the line card, SKU LC-XGS-16, errors cleared.",
        expected={
            "fault_reference": "1011",
            "reported_issue_summary": "OLT uplink line card faulting with CRC errors.",
            "reported_issue_correct_flag": True,
            "valid_issue_flag": True,
            "rc_hl_category": "Hardware",
            "rc_ll_category": "Line Card Fault",
            "rc_fault_story_verbose": "The uplink line card in the OLT chassis was generating CRC errors. The line card was swapped for SKU LC-XGS-16 and the errors cleared.",
            "faulty_components": [{"item": "Line card", "sku": "LC-XGS-16"}],
            "used_components": [{"item": "Line card", "sku": "LC-XGS-16"}],
        },
    ),
    EvalSample(
        id="flt-1012",
        raw_text="fault one thousand twelve intermittent drops traced to a bent fibre in the customer riser, replaced the drop cable sku drop dash g six five seven a two and refitted, stable",
        corrected_text="Fault 1012, intermittent drops traced to a bent fibre in the customer riser. Replaced the drop cable, SKU DROP-G657A2, and refitted, stable.",
        expected={
            "fault_reference": "1012",
            "reported_issue_summary": "Intermittent drops caused by a bent fibre in the customer riser.",
            "reported_issue_correct_flag": True,
            "valid_issue_flag": True,
            "rc_hl_category": "Physical Plant",
            "rc_ll_category": "Fibre Bend",
            "rc_fault_story_verbose": "Intermittent service drops were traced to a bent fibre in the customer riser exceeding the minimum bend radius. The drop cable was replaced with SKU DROP-G657A2 and refitted, after which the link was stable.",
            "faulty_components": [{"item": "Drop cable", "sku": "DROP-G657A2"}],
            "used_components": [{"item": "Drop cable", "sku": "DROP-G657A2"}],
        },
    ),
    EvalSample(
        id="flt-1013",
        raw_text="ticket ten thirteen o n u would not register on the p o n, o m c i not coming up, found duplicate serial in the o l t, removed stale entry and re provisioned, registered ok",
        corrected_text="Ticket 1013, ONU would not register on the PON, OMCI not coming up. Found duplicate serial in the OLT, removed stale entry and re-provisioned, registered OK.",
        expected={
            "fault_reference": "1013",
            "reported_issue_summary": "ONU failed to register on the PON due to a duplicate serial in the OLT.",
            "reported_issue_correct_flag": True,
            "valid_issue_flag": True,
            "rc_hl_category": "Configuration",
            "rc_ll_category": "Provisioning Error",
            "rc_fault_story_verbose": "The ONU would not register on the PON and OMCI was not coming up. A duplicate serial number was found in the OLT provisioning. The stale entry was removed and the ONU was re-provisioned, after which it registered successfully.",
            "faulty_components": [],
            "used_components": [],
        },
    ),
    EvalSample(
        id="flt-1014",
        raw_text="fault one oh one four, whole street off, digger cut the main feeder duct, emergency splice done on two hundred and eighty eight fibre cable, used closure sku clo dash two eight eight",
        corrected_text="Fault 1014, whole street off. Digger cut the main feeder duct. Emergency splice done on 288-fibre cable, used closure SKU CLO-288.",
        expected={
            "fault_reference": "1014",
            "reported_issue_summary": "Whole street off after a digger cut the main feeder duct.",
            "reported_issue_correct_flag": True,
            "valid_issue_flag": True,
            "rc_hl_category": "Physical Plant",
            "rc_ll_category": "Cable Damage",
            "rc_fault_story_verbose": "An excavator cut the main feeder duct, taking an entire street off service. An emergency splice was carried out on the 288-fibre cable using closure SKU CLO-288, restoring service.",
            "faulty_components": [{"item": "288-fibre feeder cable", "sku": "CLO-288"}],
            "used_components": [{"item": "Fibre closure", "sku": "CLO-288"}],
        },
    ),
    EvalSample(
        id="flt-1015",
        raw_text="job ten fifteen customer says intermittent but everything green and stable for two hours of monitoring, advised monitor further, no fault at this time",
        corrected_text="Job 1015, customer says intermittent but everything green and stable for two hours of monitoring. Advised monitor further, no fault at this time.",
        expected={
            "fault_reference": "1015",
            "reported_issue_summary": "Reported intermittent issue but link stable across two hours of monitoring.",
            "reported_issue_correct_flag": False,
            "valid_issue_flag": False,
            "rc_hl_category": "Network",
            "rc_ll_category": "No Fault Found",
            "rc_fault_story_verbose": "The customer reported an intermittent issue, but monitoring showed all indicators green and the link stable for two hours. No fault was found at this time and the customer was advised to monitor further.",
            "faulty_components": [],
            "used_components": [],
        },
    ),
    EvalSample(
        id="flt-1016",
        raw_text="fault one thousand sixteen, o n t power supply dead, no lights at all, swapped the p s u sku psu dash twelve v dash two a and o n t booted, service fine",
        corrected_text="Fault 1016, ONT power supply dead, no lights at all. Swapped the PSU, SKU PSU-12V-2A, and ONT booted, service fine.",
        expected={
            "fault_reference": "1016",
            "reported_issue_summary": "ONT dead with no lights due to a failed power supply.",
            "reported_issue_correct_flag": True,
            "valid_issue_flag": True,
            "rc_hl_category": "Power",
            "rc_ll_category": "PSU Failure",
            "rc_fault_story_verbose": "The ONT was completely dead with no lights, indicating a power problem. The power supply unit was found dead and replaced with SKU PSU-12V-2A, after which the ONT booted and service was fine.",
            "faulty_components": [{"item": "PSU", "sku": "PSU-12V-2A"}],
            "used_components": [{"item": "PSU", "sku": "PSU-12V-2A"}],
        },
    ),
]


def load_samples() -> list[EvalSample]:
    return list(SAMPLES)
