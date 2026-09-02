# Cyclone DDS profiles for the wireless link

`cyclonedds_<device>.xml` is selected by `CYCLONEDDS_URI` in
`config/devices/<device>.env`. The files hold real LAN addresses, so they are
**gitignored** — recreate them per machine from the template below.

Two things they do, both necessary:

- **`AllowMulticast=false` + explicit `<Peers>`.** WiFi APs rate-limit or drop
  multicast, and WSL's mirrored networking mode is shakiest exactly there.
  Unicast discovery removes the whole failure class.
- **Pinned `<NetworkInterface>`.** Both machines have several: the desktop has
  `eth1` (LAN), a mirrored Tailscale adapter and `docker0`; the Jetson has
  `wlP1p1s0` (WiFi), `l4tbr0` and `docker0`. Autodetection is a coin flip.

Current fleet, ROS_DOMAIN_ID 99 (RTPS ports 32150-32399):

| device | interface | address |
|---|---|---|
| desktop | `eth1`      | 192.168.50.156 |
| jetson  | `wlP1p1s0`  | 192.168.50.50  |

Give both a DHCP reservation on the router — a lease change breaks the peer
list silently, with no error, just an empty `ros2 topic list`.

```xml
<?xml version="1.0" encoding="UTF-8" ?>
<CycloneDDS xmlns="https://cdds.io/config">
  <Domain id="any">
    <General>
      <Interfaces><NetworkInterface name="IFACE"/></Interfaces>
      <AllowMulticast>false</AllowMulticast>
    </General>
    <Discovery>
      <ParticipantIndex>auto</ParticipantIndex>
      <MaxAutoParticipantIndex>32</MaxAutoParticipantIndex>
      <Peers>
        <Peer address="192.168.50.156"/>
        <Peer address="192.168.50.50"/>
      </Peers>
    </Discovery>
  </Domain>
</CycloneDDS>
```
