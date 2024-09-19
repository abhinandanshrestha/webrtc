# Use the official Postgres image as a base
FROM postgres

# Install nano
RUN apt-get update && apt-get install -y nano

# Set the default command to run Postgres
CMD ["postgres"]
