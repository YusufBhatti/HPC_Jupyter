echo "Waiting for 5 seconds..."
#sleep 5


echo "Connect to localhost"
ssh -N -J ybhatti@int5-pub.snellius.surf.nl ybhatti@tcn461.local.snellius.surf.nl -L 51525:localhost:51525
echo

# Use sshpass to handle the password
#sshpass -p "$SSH_PASS" ssh -N -J ybhatti@int5-pub.snellius.surf.nl ybhatti@tcn1104.local.snellius.surf.nl -L 51525:localhost:51525

#ssh -N -J ybhatti@int5-pub.snellius.surf.nl ybhatti@tcn1104.local.snellius.surf.nl -L 51525:localhost:51525
echo "connected to local host ${USER}@${LOGIN_HOST} ${USER}@${BATCH_HOST} -L ${PORT}:localhost:${PORT}"

