#!/bin/bash

script_dir="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
script_name="$( basename -- "${BASH_SOURCE[0]}" )"
invocation_path="$0"
source "$script_dir"/bash_utils.sh || exit 254

CLASS_NVME_MAGIC=0x010802
SYSFS_PCI_DEVICES_DIR="/sys/bus/pci/devices"
SYSFS_PCI_DRIVER_DIR="/sys/bus/pci/drivers"
NVME_DEVICE_DIR="/dev"

print_help() {
    printerr "Useage: %s:\n" "$invocation_path"
    printerr "PCI bus scanner for block devices\n"
    printerr "  -h                      print help, this message\n"
    printerr "  -l <OPT>[DEV_SLOT]      list target device properties\n"
    printerr "  -u [DEV_SLOT]           unbind target device at slot\n"
    printerr "  -b [DEV_SLOT]=[DRIVER]  bind target device at slot to be using target driver\n"
    printerr "  -m [DEV_SLOT]           find the corresponding nvme device that the target device mapped to\n"
    printerr "Default behavior\n"
    printerr "  When no option is supplied, list all block devices properties\n"
    printerr "Return values\n"
    printerr "  0   script terminates correctly\n"
    printerr "  1   invalid options\n"
    printerr "  2   target device is not found\n"
    printerr "  3   target driver is not found\n"
    printerr "  4   map call cannot find corresponding device\n"
    printerr "  5   map call can find corresponding device, but target is not using nvme driver\n"
    printerr "  254 dependency error\n"
    printerr "  255 internal error\n"
}

dup_func_detect_exit() {
    if [[ -z $functionaility ]]; then return; fi
    printf "Duplicated functionaility detected, was %s\n" "$functionaility"
    exit 1
}

functionaility=""
dev_target=""
bind_driver=""
targets=()
while getopts "hl:u:b:m:" arg; do case $arg in
    h)  print_help
        exit 0
    ;;
    l)  dup_func_detect_exit
        dev_target="${OPTARG}"
        targets=( "$SYSFS_PCI_DEVICES_DIR/$dev_target" )
        functionaility="list"
    ;;
    u)  dup_func_detect_exit
        dev_target="${OPTARG}"
        targets=( "$SYSFS_PCI_DEVICES_DIR/$dev_target" )
        functionaility="unbind"
    ;;
    b)  dup_func_detect_exit
        IFS="=" read -ra options <<< "${OPTARG}"
        [ "${#options[@]}" -eq 2 ]; assert_zero_exit 1 $? "Bind option <%s> is ill-formatted, see help" "${OPTARG}"
        dev_target="${options[0]}"
        bind_driver="${options[1]}"
        targets=( "$SYSFS_PCI_DEVICES_DIR/$dev_target" )
        functionaility="bind"
    ;;
    m)  dup_func_detect_exit
        dev_target="${OPTARG}"
        targets=( "$SYSFS_PCI_DEVICES_DIR/$dev_target" )
        functionaility="map"
    ;;
    *)  print_help
        exit 1
    ;;
esac done
if [[ -z $functionaility ]]; then functionaility="list"; fi
for sysfs_pci_dev_root in "${targets[@]}"; do
    [[ -e $sysfs_pci_dev_root ]]; assert_zero_exit 1 $? "Device to query <%s> does not exist" "$sysfs_pci_dev_root"
done
if [[ "${#targets[@]}" -eq 0 ]]; then targets=( "$SYSFS_PCI_DEVICES_DIR"/* ); fi

unbind_dev() {
    # $1 device properties
    [ "$#" -eq 1 ]; assert_zero_exit 255 $? "unbind_dev is called with invalid args"
    local -n dev_prop=$1
    local device_slot="${dev_prop["Slot"]}"
    local driver_name="${dev_prop["Driver"]}"
    printerr "Unbind device at slot %s currently using driver <%s>\n" "$device_slot" "${dev_prop["Driver"]}"
    echo -n "$device_slot" | sudo tee "$SYSFS_PCI_DRIVER_DIR/$driver_name/unbind" > /dev/null
    printerr "Unbind complete\n"
}

bind_dev() {
    # $1 device properties
    # $2 target pci driver
    [ "$#" -eq 2 ]; assert_zero_exit 255 $? "bind_dev is called with invalid args"
    local -n dev_prop=$1
    local device_slot="${dev_prop["Slot"]}"
    local driver_name="$2"
    printerr "Bind device at slot %s to use driver <%s>\n" "${dev_prop["Slot"]}" "$driver_name"
    echo -n "$device_slot" | sudo tee "$SYSFS_PCI_DRIVER_DIR/$driver_name/bind" > /dev/null
    printerr "Bind complete\n"
}

get_driver_for_pci_dev() {
    # $1 device sysfs root
    [ "$#" -eq 1 ]; assert_zero_exit 255 $? "get_driver_for_pci_dev is called with invalid args"
    local dev_root="$1"
    if [ -d "$dev_root/driver" ]; then
        basename -- "$(readlink -f "$dev_root/driver")"
    fi
}

nvme_fill_aux_properties() {
    local -n dev_prop=$1

    dev_pci_id="${dev_prop["Slot"]}"
    dev_driver="${dev_prop["Driver"]}"
    if [ "$dev_driver" == "nvme" ]; then
        dev_nvme_device_dir="$SYSFS_PCI_DEVICES_DIR/$dev_pci_id/nvme"
        if [ -d "$dev_nvme_device_dir" ]; then
            dev_nvme_device_names=()
            for dev in "$dev_nvme_device_dir"/*; do
                dev_nvme_device_names+=( "$NVME_DEVICE_DIR/$(basename --  "$dev")n1" )
            done
            printf -v nvme_dev_list_concat "%s " "${dev_nvme_device_names[@]}"
            printf -v aux_info_str "NVMe device: %s\n%s" \
                "$nvme_dev_list_concat" "$(echo "$nvme_dev_list_concat" | xargs lsblk )"
        else
            printf -v aux_info_str \
                "Internal Error: Driver specific property not found (Dir %s not found)" \
                "$dev_nvme_device_dir"
        fi
    fi

    dev_prop["NVMeDev"]=$(echo "$nvme_dev_list_concat" | xargs)
    dev_prop["NVMeAuxInfo"]="$aux_info_str"
}

if [[ $functionaility == "bind" ]]; then
    driver_exist=0
    for sysfs_driver_path in "$SYSFS_PCI_DRIVER_DIR"/*; do
        driver_name="$(basename -- "$sysfs_driver_path")"
        [[ $driver_name == "$bind_driver" ]] && driver_exist=1
    done
    assert_exit 3 "$driver_exist" \
        "Driver <%s> specified in bind operation does not exist. List of available drivers can be found in %s" \
        "$bind_driver" "$SYSFS_PCI_DRIVER_DIR"
fi

for sysfs_pci_dev_root in "${targets[@]}"; do
    dev_pci_id="$(basename -- "$sysfs_pci_dev_root")"
    dev_class="$(cat "$sysfs_pci_dev_root/class")"
    if [ "$dev_class" == "$CLASS_NVME_MAGIC" ]; then
        declare -A pci_dev_property
        as_associative_arr pci_dev_property "$(lspci -vmm -s "$dev_pci_id")" :
        pci_dev_property["Slot"]="$dev_pci_id"
        pci_dev_property["Driver"]="$(get_driver_for_pci_dev "$sysfs_pci_dev_root")"

        case $functionaility in
            bind)
                if [ "$dev_target" == "${pci_dev_property["Slot"]}" ]; then
                    [ "$bind_driver" != "${pci_dev_property["Driver"]}" ]; assert_zero_exit 255 $? \
                        "Device at slot %s is already using driver <%s>" "$dev_target" "$bind_driver"

                    if [ -n "${pci_dev_property["Driver"]}" ]; then unbind_dev pci_dev_property; fi
                    bind_dev pci_dev_property "$bind_driver"

                    as_associative_arr pci_dev_property "$(lspci -vmm -s "$dev_target")" :
                    printerr "Device at slot %s is now using driver <%s>\n" \
                        "$dev_target" "$(get_driver_for_pci_dev "$sysfs_pci_dev_root")"
                    exit 0
                fi
            ;;
            unbind)
                if [ "$dev_target" == "${pci_dev_property["Slot"]}" ]; then
                    [ -n "${pci_dev_property["Driver"]}" ]; assert_zero_exit 255 $? \
                        "Device at slot %s is already not bind to any driver" "$dev_target"

                    unbind_dev pci_dev_property

                    as_associative_arr pci_dev_property "$(lspci -vmms "$dev_target")" :
                    printerr "Device at slot %s is now using driver <%s>\n" \
                        "$dev_target" "$(get_driver_for_pci_dev "$sysfs_pci_dev_root")"
                    exit 0
                fi
            ;;
            list)
                aux_info_str=""
                dev_pci_id="${pci_dev_property["Slot"]}"
                dev_driver="${pci_dev_property["Driver"]}"
                if [ "$dev_driver" == "nvme" ]; then
                    nvme_fill_aux_properties pci_dev_property
                    aux_info_str="${pci_dev_property["NVMeAuxInfo"]}"
                fi
                if [ -n "$aux_info_str" ]; then
                    aux_info_str="    ${aux_info_str//$'\n'/$'\n'    }"$'\n'""
                fi
                printf "Slot: %s    Dev: %s -- %s\n  Driver: %s\n%s\n" \
                    "$dev_pci_id" "${pci_dev_property["Vendor"]}" "${pci_dev_property["Device"]}" \
                    "$dev_driver" "$aux_info_str"
            ;;
            map)
                if [ "$dev_target" == "${pci_dev_property["Slot"]}" ]; then
                    if [ "${pci_dev_property["Driver"]}" == "nvme" ]; then
                        nvme_fill_aux_properties pci_dev_property
                        printf "%s" "${pci_dev_property["NVMeDev"]}"
                        exit 0
                    else
                        exit 5
                    fi
                fi
            ;;
            *) assert_exit 1 "Invalid bind option %s" "$functionaility" ;;
        esac
    fi
done

if [ "$functionaility" == bind ] || [ "$functionaility" == unbind ]; then
    printerr "Target device %s not found, %s not successful\n" "$dev_target" "$functionaility"
    exit 2
fi

if [ "$functionaility" == map ]; then
    exit 4
fi
